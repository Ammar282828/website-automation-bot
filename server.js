require('dotenv').config();

const express      = require('express');
const multer       = require('multer');
const path         = require('path');
const fs           = require('fs');
const os           = require('os');
const { execFile } = require('child_process');
const sharp        = require('sharp');
const { GoogleGenAI } = require('@google/genai');
const OpenAI       = require('openai');

process.on('uncaughtException', (err) => {
    console.error('[Uncaught Exception]', err.message, err.stack);
});
process.on('unhandledRejection', (reason) => {
    console.error('[Unhandled Rejection]', reason);
});

const PORT = process.env.PORT || 3000;

// ── Provider clients ───────────────────────────────────────────────────────
const geminiClient = process.env.GOOGLE_API_KEY ? new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY }) : null;
const openaiClient = process.env.OPENAI_API_KEY  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

// Which providers are available
const PROVIDERS = {};
if (geminiClient) PROVIDERS.gemini    = { label: 'Gemini',          canGenerate: true };
if (openaiClient) PROVIDERS.openai    = { label: 'OpenAI',          canGenerate: true };
if (geminiClient) PROVIDERS.nanobana2 = { label: 'Nano Banana 2',   canGenerate: true };

console.log('[Providers]', Object.keys(PROVIDERS).join(', ') || 'NONE — add API keys to .env');

// ── Cost tracking ─────────────────────────────────────────────────────────
const COST_PER_IMAGE = {
    gemini:    0.134,   // Nano Banana Pro @ 1K
    nanobana2: 0.101,   // Nano Banana 2 @ 2K
    openai:    0.133,   // GPT Image 1.5 High @ 1024x1024
};

const usageStats = {
    session: {
        gemini:    { images: 0, cost: 0 },
        nanobana2: { images: 0, cost: 0 },
        openai:    { images: 0, cost: 0 },
        total:     { images: 0, cost: 0 },
    },
    history: [],  // last 50 entries
};

function trackUsage(provider, shotId) {
    const cost = COST_PER_IMAGE[provider] || 0;
    if (!usageStats.session[provider]) usageStats.session[provider] = { images: 0, cost: 0 };
    usageStats.session[provider].images++;
    usageStats.session[provider].cost += cost;
    usageStats.session.total.images++;
    usageStats.session.total.cost += cost;

    const entry = {
        provider,
        shotId,
        cost,
        timestamp: Date.now(),
    };
    usageStats.history.unshift(entry);
    if (usageStats.history.length > 50) usageStats.history.length = 50;

    console.log(`[Cost] +$${cost.toFixed(3)} (${provider}/${shotId}) — session total: $${usageStats.session.total.cost.toFixed(3)} (${usageStats.session.total.images} images)`);
    return entry;
}

// ── Shot definitions ───────────────────────────────────────────────────────
// Each shot has an id, label, category, and a function that builds the prompt
const SHOT_CATALOG = {
    // ── Ecommerce ──
    ecom_hero: {
        id: 'ecom_hero',
        label: 'Hero / Front',
        category: 'ecommerce',
        description: 'Clean front-facing product shot on pure white',
    },
    ecom_angle: {
        id: 'ecom_angle',
        label: '45° Angle',
        category: 'ecommerce',
        description: 'Three-quarter angle showing depth and dimension',
    },
    ecom_detail: {
        id: 'ecom_detail',
        label: 'Detail Close-up',
        category: 'ecommerce',
        description: 'Extreme macro of the finest detail area',
    },
    ecom_flat: {
        id: 'ecom_flat',
        label: 'Flat Lay',
        category: 'ecommerce',
        description: 'Bird\'s eye flat lay on white surface',
    },
    ecom_stand: {
        id: 'ecom_stand',
        label: 'Display Stand',
        category: 'ecommerce',
        description: 'On a branded display stand with warm backdrop',
    },
    ecom_group: {
        id: 'ecom_group',
        label: 'Scale / Context',
        category: 'ecommerce',
        description: 'Jewelry next to a subtle size reference',
    },

    // ── Model ──
    model_wrist: {
        id: 'model_wrist',
        label: 'Wrist / Hand',
        category: 'model',
        description: 'Jewelry on wrist or hand, tight crop',
    },
    model_neck: {
        id: 'model_neck',
        label: 'Neck / Décolletage',
        category: 'model',
        description: 'Necklace on neck, collarbone framing',
    },
    model_ear: {
        id: 'model_ear',
        label: 'Ear Close-up',
        category: 'model',
        description: 'Earring on ear, jawline framing',
    },
    model_lifestyle: {
        id: 'model_lifestyle',
        label: 'Lifestyle',
        category: 'model',
        description: 'Model wearing jewelry in lifestyle context',
    },

    // ── Marble / Surface ──
    marble: {
        id: 'marble',
        label: 'Marble Surface',
        category: 'marble',
        description: 'Luxury marble surface with soft props',
    },
    marble_dark: {
        id: 'marble_dark',
        label: 'Dark Marble',
        category: 'marble',
        description: 'Moody dark marble with dramatic lighting',
    },
};

// ── Prompt builders per shot ───────────────────────────────────────────────
function buildShotPrompt(shotId, customInstruction, hasAnchor = false) {
    const base = 'You are generating product photography for House of Mina (houseofmina.store), a luxury South Asian jewelry brand. Their aesthetic is warm, elegant, and editorial — rich gold tones, deep jewel colors, and a regal yet modern sensibility.\n\nCopy the jewelry from the reference photo(s) with absolute fidelity. Reproduce every stone, every metal tone, every proportion, every surface texture exactly. Do not add, remove, merge, or alter any design element. The generated image must be indistinguishable from a real photograph of this exact piece.';

    // When an anchor reference is present, add IP-Adapter-style consistency conditioning
    const anchorBlock = hasAnchor
        ? `\nCONSISTENCY ANCHOR: The LAST reference image is a clean studio product shot I already generated of this exact jewelry piece. Treat it as your visual ground truth. Every stone count, every prong, every metal tone, every proportion in your output MUST match this anchor image exactly. If there is any ambiguity between the raw reference photos and the anchor, defer to the anchor — it is the canonical representation of this piece.\n`
        : '';

    const scenes = {
        ecom_hero: `SCENE: Professional ecommerce hero shot. Pure white (#FFFFFF) seamless background. The jewelry is centered, occupying approximately 65% of the frame. Camera is at a slight elevation (15–20°) to show the decorative face. Even, diffused studio lighting from two softboxes at 45° angles, creating clean specular highlights on metal surfaces and brilliant stone reflections. Subtle drop shadow beneath the piece for grounding. No props, no distractions — the piece is the entire composition.
CAMERA: 100mm macro lens, f/8, focus-stacked for edge-to-edge sharpness. Color-accurate white balance (5500K). Shot on medium format digital for maximum detail.`,

        ecom_angle: `SCENE: Three-quarter angle product shot. Pure white (#FFFFFF) seamless background. Camera positioned at 45° to the front face, slightly elevated (20–25°), revealing the depth, profile, and side construction of the piece. This angle shows how the jewelry looks in three dimensions — the curve of a bangle, the height of a setting, the thickness of metalwork. Same even studio lighting with clean highlights.
CAMERA: 100mm macro, f/8, focus-stacked. The viewer should feel they can reach in and pick up the piece.`,

        ecom_detail: `SCENE: Extreme macro close-up. Camera is 1–3 cm from the most intricate area of the jewelry — the center stone and its setting, the finest filigree, or the most detailed metalwork. Fill the entire frame with this detail. White or neutral surface beneath. Razor-sharp focus on the subject with natural bokeh softening the edges. This shot reveals craftsmanship — individual prongs, stone facets, metal grain, pavé precision.
CAMERA: Dedicated macro lens at 1:1 magnification, f/5.6 for shallow depth, ring light for even illumination without harsh shadows. Shot so close the viewer can count individual stones.`,

        ecom_flat: `SCENE: Overhead flat lay on pure white surface. Camera directly above (90° bird's eye). The jewelry is laid flat, centered, with its decorative face pointing up. For bangles/bracelets: circular shape fully visible. For necklaces: arranged in an elegant drape or gentle curve. For rings: face up, slightly angled. Even, shadowless lighting from a large overhead softbox. Clean, minimal, editorial.
CAMERA: 85mm, f/8, tripod-mounted directly overhead. Perfect symmetry in composition.`,

        ecom_stand: `SCENE: House of Mina brand display presentation. Look at the reference photo(s) to determine the jewelry type, then choose the CORRECT display:
- Bangles / cuffs / bracelets: upright on a velvet cushion roll or half-cylinder stand, resting naturally with the decorative face toward camera. NEVER use a T-bar or hanging stand for bangles.
- Rings: on a slim velvet cone or small cushion, tilted slightly toward camera.
- Necklaces / chokers: draped over a fabric neck bust or laid on a velvet tray in an elegant curve.
- Earrings: on a small padded earring card or low T-bar stand.
- Maang tikka / headpieces: laid flat on a velvet tray or silk fabric.

The stand/display is elegant and minimal, in matte cream, soft gold, or deep velvet. Background is warm ivory/cream with a hint of texture (linen or fine paper). Soft, warm window-style light from the upper left creates gentle shadows and a luxurious mood. The decorative face of the jewelry faces the camera. The display should look natural — the jewelry should sit the way it would in a real boutique.
CAMERA: 85mm f/2.8, slightly shallow depth of field to separate the piece from the background. Warm color temperature (5800K).`,

        ecom_group: `SCENE: Scale and context shot. The jewelry is placed on a pure white surface alongside a subtle, universally understood size reference — a single fresh rose petal, a small velvet pouch, or an elegant hand mirror. The reference object is secondary and slightly out of focus. The jewelry remains the hero. This shot communicates real-world scale and presence.
CAMERA: 85mm, f/4, with the jewelry in sharp focus and the reference object in soft focus behind or beside it.`,

        model_wrist: `SCENE: The jewelry is worn on the wrist/hand of a model. For bangles and bracelets: worn snugly on the wrist, sitting flush against skin with the outer decorative face toward the camera. For rings: worn on the ring finger, hand relaxed.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, natural skin texture, clean manicure with nude or soft pink nails
- Arm extended forward, elbow slightly bent
- Wrist level or slightly lowered, fingers pointing DOWNWARD and loosely relaxed
- Back of hand faces the camera, palm faces away
- Do NOT raise the hand with fingers pointing up, do NOT show the palm

FRAMING: Tight crop showing only the hand, wrist, and a few inches of forearm. No face, no shoulder, no torso.
LIGHTING: Single soft key light from above-left, warm neutral blurred backdrop (creamy beige or soft gold), 85mm f/1.4 equivalent depth of field. Subtle film grain. The skin should glow warmly.`,

        model_neck: `SCENE: The jewelry is worn around the neck of a model. The necklace or choker sits naturally on the collarbone/décolletage area.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, natural skin, elegant bone structure
- Head tilted very slightly to one side, chin slightly lifted
- Wearing a simple, solid-color top or bare shoulders (nothing competing with the jewelry)
- Hair pulled back or swept to one side to fully reveal the necklace

FRAMING: From mid-chest to just below the chin. The necklace is the clear focal point. Jawline and neck visible for context but the jewelry dominates.
LIGHTING: Soft, warm key light from above-right, gentle fill from the left. Warm neutral backdrop. 85mm f/1.8, shallow depth. The skin glows, the metal catches light beautifully.`,

        model_ear: `SCENE: The earring is worn on the ear of a model. Close-up of the ear, jawline, and a hint of neck.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, clean skin, elegant jawline
- Head turned slightly (three-quarter profile) to present the ear naturally
- Hair tucked behind the ear or swept up to fully reveal the earring
- Expression serene, mouth relaxed (if lips are visible at edge of frame)

FRAMING: Tight crop on the ear and surrounding area. The earring is the clear hero. Show enough of the jaw and neck for anatomical context.
LIGHTING: Soft key light from the front-left, gentle rim light to separate from background. Warm blurred backdrop. 100mm f/2, very shallow depth — the earring is razor-sharp, everything else falls off softly.`,

        model_lifestyle: `SCENE: Lifestyle editorial shot. The model is wearing the jewelry in a warm, luxurious setting — think golden hour light, soft furnishings, or a beautiful window. The mood is aspirational, elegant, and distinctly South Asian-luxe.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, styled beautifully but not overly made up
- Natural, candid-feeling pose — adjusting the jewelry, looking away from camera, or mid-movement
- Wearing complementary but simple clothing that doesn't compete (solid colors, elegant draping)

FRAMING: Medium shot (waist up or three-quarter). The jewelry should be clearly visible and prominent despite the wider framing. Environmental context adds mood without overwhelming.
LIGHTING: Warm, natural-feeling light (golden hour or large window). Slight haze or warmth in the atmosphere. 50mm f/1.8, cinematic depth of field. Slight film grain for editorial feel.`,

        marble: `SCENE: Luxury surface shot. The jewelry rests on a white or cream Carrara marble surface with soft, natural grey veining. Beside the jewelry (not touching): one or two minimal props — a small sprig of dried flowers, a fragment of silk ribbon, or a tiny gold-rimmed dish. Props are muted and secondary. The composition is editorial, airy, and luxurious.
LIGHTING: Soft, warm natural light from a large window to the left. Gentle shadows. The marble surface has a slight sheen. Warm color palette overall.
CAMERA: 45° angle, 85mm f/2.8, the jewelry is in perfect focus, props and marble veining fall off softly. The feeling is a luxury magazine editorial spread.`,

        marble_dark: `SCENE: Dramatic dark surface shot. The jewelry rests on dark emperador or nero marquina marble — deep brown-black with gold or white veining. The mood is dramatic, moody, and high-end. Minimal props if any — perhaps a single dark velvet fold or a matte black box edge barely visible. The jewelry catches all the light and pops against the dark surface.
LIGHTING: Single focused light source from above-right, creating dramatic highlights on the metal and stones while the marble stays dark and moody. Deep shadows, high contrast. The gold of the jewelry glows against the darkness.
CAMERA: Low angle (15–20°), 100mm f/2.8, shallow depth. Cinematic, editorial, powerful. Think luxury brand campaign.`,
    };

    const scene = scenes[shotId] || scenes.ecom_hero;

    const parts = [
        base,
        anchorBlock,
        scene,
        '',
        'OUTPUT: Square 1:1 aspect ratio. Photorealistic — indistinguishable from a real photograph. No AI artifacts, no floating elements, no impossible reflections.',
        ...(customInstruction ? [`\nADDITIONAL DIRECTION: ${customInstruction}`] : []),
    ];

    return parts.join('\n');
}

const app    = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.use(express.json({ limit: '100mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// ── Serve available providers + shot catalog to frontend ────────────────────
app.get('/providers', (_req, res) => res.json(PROVIDERS));
app.get('/shots', (_req, res) => res.json(SHOT_CATALOG));
app.get('/usage', (_req, res) => res.json(usageStats));
app.post('/usage/reset', (_req, res) => {
    for (const key of Object.keys(usageStats.session)) {
        usageStats.session[key] = { images: 0, cost: 0 };
    }
    usageStats.history = [];
    res.json({ reset: true });
});
app.get('/cost-rates', (_req, res) => res.json(COST_PER_IMAGE));

// ── Serve generated files from disk ─────────────────────────────────────────
app.get('/file', (req, res) => {
    const filePath = req.query.path;
    if (!filePath || !fs.existsSync(filePath)) return res.status(404).send('Not found');
    res.sendFile(path.resolve(filePath));
});

// ── Batch cancellation state ────────────────────────────────────────────────
let activeBatchId  = null;
let batchCancelled = false;

const cancelHandler = (req, res) => {
    if (activeBatchId) {
        batchCancelled = true;
        res.json({ cancelled: true });
    } else {
        res.json({ cancelled: false, message: 'No active batch.' });
    }
};
app.post('/cancel-batch', cancelHandler);
app.post('/batch/cancel', cancelHandler);

// ── Single product generation ───────────────────────────────────────────────
app.post('/generate', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const shotIds           = JSON.parse(req.body.shots || '[]');
    const customInstruction = (req.body.customInstruction || '').trim() || null;
    const provider          = (req.body.provider || 'gemini').trim();

    if (shotIds.length === 0) return res.status(400).json({ error: 'No shots selected.' });

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        console.log(`[Generate] ${shotIds.length} shot(s) via ${provider}: ${shotIds.join(', ')}`);

        // ── Anchor-first consistency pipeline ──
        // Always generate a clean hero/product shot first as the visual anchor.
        // This anchor is then fed as a reference into EVERY subsequent shot
        // (ecom, model, marble — all of them), similar to how ComfyUI's
        // IP-Adapter conditions all generations from a single reference.

        const results = [];
        let anchorRef = null;

        // Determine anchor: use ecom_hero if selected, otherwise first ecom shot, otherwise first shot
        const anchorId = shotIds.includes('ecom_hero') ? 'ecom_hero'
            : shotIds.find(id => id.startsWith('ecom_'))
            || shotIds[0];

        // Generate anchor shot (no anchor reference for the anchor itself)
        console.log(`[Anchor] Generating ${anchorId} as consistency anchor via ${provider}...`);
        const anchorShot = SHOT_CATALOG[anchorId];
        const anchorData = await generateShot(anchorId, imageInputs, customInstruction, false, provider);
        anchorRef = { base64: anchorData, mimeType: 'image/png' };
        results.push({ id: anchorId, label: anchorShot.label, category: anchorShot.category, data: anchorData });

        // Generate all remaining shots in parallel, ALL receiving the anchor
        const remaining = shotIds.filter(id => id !== anchorId);

        if (remaining.length > 0) {
            const refsWithAnchor = [...imageInputs, anchorRef];
            const parallel = await Promise.all(remaining.map(async (shotId) => {
                const shot = SHOT_CATALOG[shotId];
                if (!shot) return null;
                const data = await generateShot(shotId, refsWithAnchor, customInstruction, true, provider);
                return { id: shotId, label: shot.label, category: shot.category, data };
            }));
            results.push(...parallel.filter(Boolean));
        }

        // Sort results in the order they were requested
        const ordered = shotIds.map(id => results.find(r => r.id === id)).filter(Boolean);

        res.json({ success: true, results: { shots: ordered }, usage: usageStats });
    } catch (err) {
        console.error('[Generate Error]', err?.message || err);
        const safetyBlocked = err?.message?.toLowerCase().includes('safety');
        res.status(500).json({
            error: safetyBlocked
                ? 'Image blocked by safety filters — try a different photo.'
                : err.message || 'Generation failed.',
        });
    }
});

// ── Retry single shot ───────────────────────────────────────────────────────
app.post('/generate-angle', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const shotId            = req.body.angleId;
    const customInstruction = (req.body.customInstruction || '').trim() || null;
    const provider          = (req.body.provider || 'gemini').trim();

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        const shot = SHOT_CATALOG[shotId];
        if (!shot) return res.status(400).json({ error: 'Unknown shot type.' });

        const imageData = await generateShot(shotId, imageInputs, customInstruction, false, provider);
        res.json({ success: true, imageData, usage: usageStats });
    } catch (err) {
        console.error('[Retry Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Generation failed.' });
    }
});

// ── Batch folder endpoint (SSE) ─────────────────────────────────────────────
app.get('/batch', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const send = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`);

    const folderPath        = (req.query.folderPath || '').trim().replace(/^['"]|['"]$/g, '');
    const customInstruction = (req.query.customInstruction || '').trim() || null;
    const shotIds           = JSON.parse(req.query.shots || '[]');
    const provider          = (req.query.provider || 'gemini').trim();

    if (!folderPath) { send({ type: 'error', message: 'No folder path provided.' }); return res.end(); }
    if (!fs.existsSync(folderPath)) { send({ type: 'error', message: `Folder not found: ${folderPath}` }); return res.end(); }
    if (!fs.statSync(folderPath).isDirectory()) { send({ type: 'error', message: 'That path is a file, not a folder.' }); return res.end(); }
    if (shotIds.length === 0) { send({ type: 'error', message: 'No shots selected.' }); return res.end(); }

    const productDirs = fs.readdirSync(folderPath, { withFileTypes: true })
        .filter(d => d.isDirectory() && !d.name.startsWith('.') && d.name !== 'ecommerce' && d.name !== 'output')
        .map(d => ({ name: d.name, fullPath: path.join(folderPath, d.name) }));

    if (productDirs.length === 0) {
        send({ type: 'error', message: 'No product subfolders found.' });
        return res.end();
    }

    activeBatchId = Date.now().toString();
    batchCancelled = false;

    send({ type: 'start', total: productDirs.length, batchId: activeBatchId, shots: shotIds });

    for (const { name: productName, fullPath: productFolder } of productDirs) {
        if (batchCancelled) {
            send({ type: 'cancelled', message: 'Batch cancelled by user.' });
            break;
        }

        send({ type: 'product_start', product: productName, productFolder });

        try {
            const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
            const imageFiles = fs.readdirSync(productFolder)
                .filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.'))
                .map(f => path.join(productFolder, f));

            if (imageFiles.length === 0) {
                send({ type: 'product_error', product: productName, message: 'No images found in folder.' });
                continue;
            }

            const imageInputs = await Promise.all(imageFiles.map(async (fp) => {
                const buf = await toJpeg(fp, fs.readFileSync(fp));
                return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
            }));

            const outDir = path.join(folderPath, 'output', productName);
            fs.mkdirSync(outDir, { recursive: true });

            // ── Anchor-first consistency pipeline ──
            const anchorId = shotIds.includes('ecom_hero') ? 'ecom_hero'
                : shotIds.find(id => id.startsWith('ecom_'))
                || shotIds[0];

            let anchorRef = null;
            const anchorShot = SHOT_CATALOG[anchorId];

            send({ type: 'angle_start', product: productName, angle: anchorId, label: `${anchorShot.label} (anchor)` });
            try {
                const b64 = await generateShot(anchorId, imageInputs, customInstruction, false, provider);
                anchorRef = { base64: b64, mimeType: 'image/png' };
                const outPath = path.join(outDir, `${anchorId}.png`);
                fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
                send({ type: 'angle_done', product: productName, angle: anchorId, label: anchorShot.label, savedTo: outPath });
                send({ type: 'usage', usage: usageStats });
            } catch (err) {
                send({ type: 'angle_error', product: productName, angle: anchorId, message: err.message });
            }

            if (batchCancelled) {
                send({ type: 'product_done', product: productName });
                send({ type: 'cancelled', message: 'Batch cancelled by user.' });
                break;
            }

            // All remaining shots get the anchor reference
            const remaining = shotIds.filter(id => id !== anchorId);
            const refsWithAnchor = anchorRef ? [...imageInputs, anchorRef] : imageInputs;
            const hasAnchor = !!anchorRef;

            for (const shotId of remaining) {
                const shot = SHOT_CATALOG[shotId];
                if (!shot) continue;
                send({ type: 'angle_start', product: productName, angle: shotId, label: shot.label });
            }

            const parallelTasks = remaining.map(shotId => {
                const shot = SHOT_CATALOG[shotId];
                if (!shot) return Promise.resolve();
                return generateShot(shotId, refsWithAnchor, customInstruction, hasAnchor, provider)
                    .then(b64 => {
                        const p = path.join(outDir, `${shotId}.png`);
                        fs.writeFileSync(p, Buffer.from(b64, 'base64'));
                        send({ type: 'angle_done', product: productName, angle: shotId, label: shot.label, savedTo: p });
                        send({ type: 'usage', usage: usageStats });
                    })
                    .catch(err => send({ type: 'angle_error', product: productName, angle: shotId, message: err.message }));
            });

            await Promise.all(parallelTasks);
        } catch (err) {
            console.error(`[Batch] ${productName}:`, err.message);
            send({ type: 'product_error', product: productName, message: err.message });
        }

        send({ type: 'product_done', product: productName });

        if (batchCancelled) {
            send({ type: 'cancelled', message: 'Batch cancelled by user.' });
            break;
        }
    }

    const wasCancelled = batchCancelled;
    activeBatchId = null;
    batchCancelled = false;

    if (!wasCancelled) send({ type: 'done' });
    res.end();
});

// ── Batch retry single shot ─────────────────────────────────────────────────
app.post('/retry-angle', upload.none(), async (req, res) => {
    const { productFolder, angleId, provider: retryProvider } = req.body;
    const provider = (retryProvider || 'gemini').trim();
    if (!productFolder || !angleId) return res.status(400).json({ error: 'Missing productFolder or angleId.' });

    const shot = SHOT_CATALOG[angleId];
    if (!shot) return res.status(400).json({ error: 'Unknown shot type.' });

    const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
    const imageFiles = fs.readdirSync(productFolder)
        .filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.'))
        .map(f => path.join(productFolder, f));

    if (imageFiles.length === 0) return res.status(400).json({ error: 'No source images in product folder.' });

    try {
        const imageInputs = await Promise.all(imageFiles.map(async (fp) => {
            const buf = await toJpeg(fp, fs.readFileSync(fp));
            return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
        }));

        const raw = await generateShot(angleId, imageInputs, null, false, provider);
        const outPath = path.join(productFolder, '..', 'output', path.basename(productFolder), `${angleId}.png`);
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, Buffer.from(raw, 'base64'));
        res.json({ success: true, base64: raw, usage: usageStats });
    } catch (err) {
        console.error('[Retry Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Retry failed.' });
    }
});

// ── WhatsApp caption generation ────────────────────────────────────────────
app.post('/generate-caption', upload.array('images[]', 10), async (req, res) => {
    const productName   = (req.body.productName || '').trim() || 'this piece';
    const extraContext   = (req.body.extraContext || '').trim();

    // Build image inputs from uploaded files (if any)
    let imageInputs = [];
    if (req.files && req.files.length > 0) {
        imageInputs = await Promise.all(req.files.map(async (f) => {
            const buf = await toJpeg(f.originalname || '', f.buffer);
            return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
        }));
    }

    // Also accept base64 images from JSON body (for generated images)
    const jsonImages = req.body.captionImages ? JSON.parse(req.body.captionImages) : [];
    for (const img of jsonImages) {
        if (img.b64) imageInputs.push({ base64: img.b64, mimeType: 'image/png' });
    }

    const captionPrompt = `You are the copywriter for House of Mina (houseofmina.store), a luxury South Asian jewelry brand based in Karachi. You write WhatsApp community posts to showcase new jewelry pieces.

BRAND VOICE & STYLE:
- Sophisticated yet accessible, warm, elegant, aspirational
- Short punchy sentences. Conversational but elevated
- Open with a hook that creates desire (e.g. "Meet your new obsession.", "Some pieces just speak for themselves.", "This one's going to turn heads.")
- Highlight the key visual feature of THIS specific piece (describe what you actually see in the image — the stone color, the design style, the sparkle)
- Use WhatsApp bold formatting with asterisks for key specs: *925 Sterling Silver*, *Gold Plated*, etc.
- End with a soft-launch / urgency line, then the standard CTA

DEFAULT MATERIAL SPECS (use these unless the image clearly shows otherwise or extra context overrides):
- 925 Sterling Silver with White Rhodium / Gold Plating
- Cubic Zirconia stones
- Simulated coloured stones (e.g. *simulated emeralds*, *simulated rubies*) — NOT certified/natural unless specified

WHATSAPP FORMATTING RULES:
- Use *asterisks* for bold (key specs, brand name)
- Use _underscores_ for italic (rare, only for emphasis)
- Line breaks between sections (hook / description / CTA)
- Emojis only at the CTA section at the end
- Keep the whole caption under 500 characters

SAMPLE FOR REFERENCE (match this energy and structure):
"Meet your new obsession. *A certified yellow sapphire. Brilliant zircon accents. 925 sterling silver*. A combination this stunning doesn't come along often — and at *House of Mina*, it's entirely yours. We're celebrating our soft launch with special introductory pricing. These pieces won't wait forever. 📩 DM to order 🇵🇰 Nationwide Delivery"

CTA BLOCK (always end with this exact block):
📩 DM to order
🇵🇰 Nationwide Delivery

${extraContext ? `EXTRA CONTEXT FROM THE USER: ${extraContext}\n` : ''}
Now look at the jewelry image(s) provided and write ONE WhatsApp community caption for ${productName}. Output ONLY the caption text, nothing else — no quotes, no explanation, no markdown code blocks.`;

    try {
        let captionText;

        if (geminiClient && imageInputs.length > 0) {
            const parts = [
                { text: captionPrompt },
                ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
            ];
            await acquireGeminiSlot();
            try {
                const response = await geminiClient.models.generateContent({
                    model: 'gemini-3.1-pro-preview',
                    contents: [{ parts }],
                });
                const resParts = response.candidates?.[0]?.content?.parts || [];
                captionText = resParts.map(p => p.text).filter(Boolean).join('').trim();
            } finally {
                releaseGeminiSlot();
            }
        } else if (geminiClient) {
            // Text-only (no images)
            await acquireGeminiSlot();
            try {
                const response = await geminiClient.models.generateContent({
                    model: 'gemini-3.1-pro-preview',
                    contents: [{ parts: [{ text: captionPrompt }] }],
                });
                const resParts = response.candidates?.[0]?.content?.parts || [];
                captionText = resParts.map(p => p.text).filter(Boolean).join('').trim();
            } finally {
                releaseGeminiSlot();
            }
        } else {
            return res.status(500).json({ error: 'No AI provider available for caption generation.' });
        }

        // Clean up: remove wrapping quotes or code blocks if the model added them
        captionText = captionText.replace(/^["'`]+|["'`]+$/g, '').replace(/^```[\s\S]*?\n/, '').replace(/\n```$/, '').trim();

        res.json({ success: true, caption: captionText });
    } catch (err) {
        console.error('[Caption Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Caption generation failed.' });
    }
});

// ── Download ZIP endpoint ───────────────────────────────────────────────────
app.post('/download-zip', async (req, res) => {
    const { images } = req.body;
    if (!images || !Array.isArray(images) || images.length === 0) return res.status(400).json({ error: 'No images.' });

    const entries = images.map((img, i) => ({
        name: img.name || `image-${i + 1}.png`,
        data: Buffer.from(img.data, 'base64'),
    }));

    const zipBuf = buildZip(entries);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', 'attachment; filename="house-of-mina-shots.zip"');
    res.send(zipBuf);
});

function buildZip(entries) {
    const localHeaders = [];
    const centralHeaders = [];
    let offset = 0;

    for (const { name, data } of entries) {
        const nameBuf = Buffer.from(name, 'utf8');
        const lh = Buffer.alloc(30);
        lh.writeUInt32LE(0x04034b50, 0);
        lh.writeUInt16LE(20, 4);
        lh.writeUInt16LE(0, 8);
        lh.writeUInt32LE(data.length, 18);
        lh.writeUInt32LE(data.length, 22);
        lh.writeUInt16LE(nameBuf.length, 26);
        localHeaders.push(Buffer.concat([lh, nameBuf, data]));

        const ch = Buffer.alloc(46);
        ch.writeUInt32LE(0x02014b50, 0);
        ch.writeUInt16LE(20, 4);
        ch.writeUInt16LE(20, 6);
        ch.writeUInt32LE(data.length, 20);
        ch.writeUInt32LE(data.length, 24);
        ch.writeUInt16LE(nameBuf.length, 28);
        ch.writeUInt32LE(offset, 42);
        centralHeaders.push(Buffer.concat([ch, nameBuf]));

        offset += 30 + nameBuf.length + data.length;
    }

    const centralBuf = Buffer.concat(centralHeaders);
    const eocd = Buffer.alloc(22);
    eocd.writeUInt32LE(0x06054b50, 0);
    eocd.writeUInt16LE(entries.length, 8);
    eocd.writeUInt16LE(entries.length, 10);
    eocd.writeUInt32LE(centralBuf.length, 12);
    eocd.writeUInt32LE(offset, 16);

    return Buffer.concat([...localHeaders, centralBuf, eocd]);
}

// ── Universal shot generator (multi-provider) ───────────────────────────────
async function generateShot(shotId, imageInputs, customInstruction, hasAnchor = false, provider = 'gemini') {
    const prompt = buildShotPrompt(shotId, customInstruction, hasAnchor);

    let result;
    if (provider === 'openai') {
        result = await generateWithOpenAI(prompt, imageInputs);
    } else if (provider === 'nanobana2') {
        result = await generateWithNanoBana2(prompt, imageInputs);
    } else {
        result = await generateWithGemini(prompt, imageInputs);
    }

    trackUsage(provider, shotId);
    return result;
}

async function generateWithGemini(prompt, imageInputs) {
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    const raw = await callGemini(parts);
    return makeSquareBase64(raw);
}

async function generateWithOpenAI(prompt, imageInputs) {
    await acquireGeminiSlot(); // reuse the same concurrency limiter
    try {
        // Use gpt-image-1.5 via the Images API with reference images
        const imageFiles = imageInputs.map((img, i) => {
            const buf = Buffer.from(img.base64, 'base64');
            return new File([buf], `ref_${i}.png`, { type: img.mimeType });
        });

        console.log(`[OpenAI] calling gpt-image-1.5... (${imageFiles.length} reference image(s))`);

        const response = await openaiClient.images.edit({
            model: 'gpt-image-1.5',
            image: imageFiles,
            prompt: prompt,
            n: 1,
            size: '1024x1024',
            quality: 'high',
        });

        const b64 = response.data?.[0]?.b64_json;
        if (!b64) {
            throw new Error('OpenAI returned no image data');
        }

        // Validate
        const buf = Buffer.from(b64, 'base64');
        const meta = await sharp(buf).metadata();
        if (!meta.width || !meta.height) throw new Error('OpenAI returned invalid image');

        console.log('[OpenAI] image OK');
        return makeSquareBase64(b64);
    } finally {
        releaseGeminiSlot();
    }
}

async function generateWithNanoBana2(prompt, imageInputs) {
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    const raw = await callNanoBana2(parts);
    return makeSquareBase64(raw);
}

// ── Shared Gemini call with retry + backoff + concurrency ───────────────────
const MAX_RETRIES = 3;
const RETRY_DELAYS = [2000, 5000, 10000];

const MAX_CONCURRENT = 3;
let activeGeminiCalls = 0;
const geminiQueue = [];

function acquireGeminiSlot() {
    return new Promise(resolve => {
        if (activeGeminiCalls < MAX_CONCURRENT) {
            activeGeminiCalls++;
            resolve();
        } else {
            geminiQueue.push(resolve);
        }
    });
}

function releaseGeminiSlot() {
    activeGeminiCalls--;
    if (geminiQueue.length > 0) {
        activeGeminiCalls++;
        geminiQueue.shift()();
    }
}

async function callGemini(parts, attempt = 0) {
    await acquireGeminiSlot();
    try {
        console.log(`[Gemini] calling... (${parts.filter(p => p.inlineData).length} image(s))${attempt > 0 ? ` [retry ${attempt}]` : ''}`);
        const response = await geminiClient.models.generateContent({
            model: 'gemini-3-pro-image-preview',
            contents: [{ parts }],
            config: { responseModalities: ['TEXT', 'IMAGE'] },
        });

        const resParts  = response.candidates?.[0]?.content?.parts || [];
        const imagePart = resParts.find(p => p.inlineData?.data && !p.thought);
        if (!imagePart) {
            const text = resParts.find(p => p.text)?.text || 'none';
            console.error('[Gemini] No image. Response text:', text.slice(0, 300));
            throw new Error('Gemini returned no image — ' + text.slice(0, 120));
        }

        try {
            const buf = Buffer.from(imagePart.inlineData.data, 'base64');
            const meta = await sharp(buf).metadata();
            if (!meta.width || !meta.height) throw new Error('Invalid image dimensions');
        } catch (valErr) {
            throw new Error('Gemini returned invalid image data');
        }

        console.log('[Gemini] image OK');
        return imagePart.inlineData.data;
    } catch (err) {
        if (attempt < MAX_RETRIES - 1) {
            const delay = RETRY_DELAYS[attempt] || 5000;
            console.log(`[Gemini] retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms...`);
            await new Promise(r => setTimeout(r, delay));
            return callGemini(parts, attempt + 1);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

async function callNanoBana2(parts, attempt = 0) {
    await acquireGeminiSlot();
    try {
        console.log(`[NanoBana2] calling gemini-3.1-flash-image-preview... (${parts.filter(p => p.inlineData).length} image(s))${attempt > 0 ? ` [retry ${attempt}]` : ''}`);
        const response = await geminiClient.models.generateContent({
            model: 'gemini-3.1-flash-image-preview',
            contents: [{ parts }],
            config: {
                responseModalities: ['TEXT', 'IMAGE'],
                imageConfig: {
                    aspectRatio: '1:1',
                    imageSize: '2K',
                },
            },
        });

        const resParts  = response.candidates?.[0]?.content?.parts || [];
        const imagePart = resParts.find(p => p.inlineData?.data && !p.thought);
        if (!imagePart) {
            const text = resParts.find(p => p.text)?.text || 'none';
            console.error('[NanoBana2] No image. Response text:', text.slice(0, 300));
            throw new Error('Nano Banana 2 returned no image — ' + text.slice(0, 120));
        }

        try {
            const buf = Buffer.from(imagePart.inlineData.data, 'base64');
            const meta = await sharp(buf).metadata();
            if (!meta.width || !meta.height) throw new Error('Invalid image dimensions');
        } catch (valErr) {
            throw new Error('Nano Banana 2 returned invalid image data');
        }

        console.log('[NanoBana2] image OK');
        return imagePart.inlineData.data;
    } catch (err) {
        if (attempt < MAX_RETRIES - 1) {
            const delay = RETRY_DELAYS[attempt] || 5000;
            console.log(`[NanoBana2] retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms...`);
            await new Promise(r => setTimeout(r, delay));
            return callNanoBana2(parts, attempt + 1);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

// ── Image helpers ───────────────────────────────────────────────────────────
async function makeSquareBase64(base64) {
    const buf = Buffer.from(base64, 'base64');
    const out = await makeSquare(buf);
    return out.toString('base64');
}

async function makeSquare(buffer) {
    const meta = await sharp(buffer).metadata();
    const size = Math.max(meta.width, meta.height);
    return sharp(buffer)
        .resize({ width: size, height: size, fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
        .png()
        .toBuffer();
}

async function toJpeg(filePathOrName, buffer) {
    const ext = path.extname(filePathOrName).toLowerCase();
    if (ext === '.heic' || ext === '.heif') {
        const tmpIn  = path.join(os.tmpdir(), `heic-in-${Date.now()}.heic`);
        const tmpOut = path.join(os.tmpdir(), `heic-out-${Date.now()}.jpg`);
        try {
            fs.writeFileSync(tmpIn, buffer);
            await new Promise((resolve, reject) => {
                execFile('sips', ['-s', 'format', 'jpeg', tmpIn, '--out', tmpOut], err => err ? reject(err) : resolve());
            });
            return fs.readFileSync(tmpOut);
        } finally {
            if (fs.existsSync(tmpIn)) fs.unlinkSync(tmpIn);
            if (fs.existsSync(tmpOut)) fs.unlinkSync(tmpOut);
        }
    }
    return sharp(buffer).jpeg({ quality: 95 }).toBuffer();
}

// ── Start ───────────────────────────────────────────────────────────────────
app.listen(PORT, () => console.log(`\nHouse of Mina Pipeline → http://localhost:${PORT}\n`));
