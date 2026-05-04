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
const USE_VERTEX = process.env.GOOGLE_GENAI_USE_VERTEXAI === 'true' || (process.env.GOOGLE_API_KEY || '').startsWith('AQ.');
const geminiClient = process.env.GOOGLE_API_KEY
    ? new GoogleGenAI({ vertexai: USE_VERTEX, apiKey: process.env.GOOGLE_API_KEY })
    : null;
if (geminiClient) console.log('[Gemini] mode:', USE_VERTEX ? 'Vertex AI (express)' : 'AI Studio');
const openaiClient = process.env.OPENAI_API_KEY  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

// Multi-brand support — drives brand-specific prompts, assets, captions, and
// overlay logos. The active brand flows in on req.body.brand / req.query.brand
// (default falls back to DEFAULT_BRAND).
const { BRANDS, DEFAULT_BRAND, resolveBrand, listBrands } = require('./brands');

// Which providers are available — only the two NanoBanana tiers are surfaced
// in the UI. OpenAI is wired but hidden — kept available as a fallback target.
const PROVIDERS = {};
if (geminiClient) PROVIDERS.gemini    = { label: 'Nano Banana Pro', tier: 'pro',  canGenerate: true, description: 'Gemini 3 Pro Image — thinking mode, highest fidelity' };
if (geminiClient) PROVIDERS.nanobana2 = { label: 'Nano Banana 2',   tier: 'fast', canGenerate: true, description: 'Gemini 3.1 Flash Image — fast and affordable' };
if (openaiClient) PROVIDERS.openai    = { label: 'OpenAI',          tier: 'hidden', canGenerate: true, hidden: true };

console.log('[Providers]', Object.keys(PROVIDERS).join(', ') || 'NONE — add API keys to .env');

// ── Cost tracking ─────────────────────────────────────────────────────────
// Per-resolution pricing (Gemini API — ai.google.dev/gemini-api/docs/pricing)
const COST_BY_SIZE = {
    gemini: {   // Nano Banana Pro (gemini-3-pro-image-preview)
        '1K': 0.134, '2K': 0.134, '4K': 0.240,
        default: 0.134,
    },
    nanobana2: { // Nano Banana 2 (gemini-3.1-flash-image-preview)
        '512': 0.034, '1K': 0.067, '2K': 0.101, '4K': 0.151,
        default: 0.067,
    },
    openai: {
        default: 0.133,  // GPT Image 1.5 High @ 1024x1024
    },
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

function trackUsage(provider, shotId, imageSize) {
    const providerCosts = COST_BY_SIZE[provider] || {};
    const cost = providerCosts[imageSize] || providerCosts.default || 0;
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

    // Brand-specific extraShots (e.g. Taheri's emerald-walnut signature) live
    // in brands.js and are merged into the catalog only when their brand is
    // active — see buildShotCatalog().
};

// ── Prompt builders per shot ───────────────────────────────────────────────
function buildShotPrompt(shotId, customInstruction, hasAnchor = false, customPrompt = null, brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    // Brand-aware preamble — fidelity rules + brand voice come from brands.js.
    const base = brand.baseIntro;

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

        ecom_stand: `SCENE: ${brand.ecomStandBrandRef}. Look at the reference photo(s) to determine the jewelry type, then choose the CORRECT display:
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

        // ── Taheri Signature Style ──
        taheri_signature: `SCENE: Taheri brand signature product photography. Study the reference photo(s) VERY carefully to determine the jewelry type and number of pieces, then follow the exact display rules below.

BACKGROUND: Dark matte emerald green background. The green fills the ENTIRE background and extends to all edges. No other colors, no gradients, no grey.

WOOD: Every jewelry item MUST be placed on an appropriate dark walnut wooden stand or display. The wood is rich brown with visible grain, smooth polished finish, warm-toned. Never painted, never black, never light/blonde wood. The wood should look premium and handcrafted. The jewelry must NEVER be placed directly on the background — it ALWAYS sits on or hangs from a wooden display piece.

DISPLAY RULES — choose EXACTLY based on what you see in the reference:

IF SINGLE RING: Place it on a small rectangular wooden block (cube or rectangular prism, roughly 4cm × 3cm × 3cm). The ring sits upright on the top edge with its decorative face angled toward camera. The block sits directly on the green velvet.

IF SINGLE PENDANT or SINGLE CHAIN: Use a smooth walnut neck bust (no head, just neck and upper chest shape). The pendant/chain hangs naturally around the bust. The bust sits on a small walnut base on the green velvet.

IF JEWELRY SET (necklace + earrings, or necklace + earrings + ring): Use a smooth walnut neck bust as the centerpiece. The necklace drapes over the bust naturally. The earrings are attached to the bust at ear level (one on each side). If a ring is included, place it on a small walnut cylinder or platform at the base of the bust. Everything is arranged symmetrically. The wooden bust sits directly on the green velvet.

IF BANGLES / BRACELETS: A vertical walnut cylinder or cone stand. The bangle rests on it showing its full circular shape and decorative face.

IF EARRINGS ONLY: A wooden earring stand — a T-bar or vertical post with a horizontal bar at the top, made of dark walnut. The earrings hang from the bar naturally, showing their full length and drop. The stand looks like a miniature clothes hanger shape in wood. NOT a flat piece of wood — it must be an actual earring display stand that the earrings hang from.

IF MAANG TIKKA / HEADPIECE: A smooth walnut dome stand. The headpiece drapes over it showing chain and pendant.

CRITICAL RULES:
- The walnut wood display must look premium, smooth, and clean — no rough edges, no imperfections
- The emerald green velvet extends to ALL edges of the frame with NO other surface or color visible
- The jewelry is the absolute HERO — sharp, well-lit, every detail visible
- The wooden display is subordinate — it supports the jewelry, never distracts from it
- All pieces in a set MUST be visible and arranged together in one composition
- The composition should feel balanced and symmetrical

LIGHTING: Warm, soft studio light from above-left with gentle fill from right. Creates warm glow on gold, clean highlights on stones, and soft shadows on the green velvet beneath the display. Color temperature 5500–5800K. The emerald background has subtle tonal variation from the directional light — slightly lighter where the key light falls, darker toward edges for a natural vignette feel.
CAMERA: Straight-on to slightly above (20–35° elevation), 85mm f/2.8, shallow depth of field — jewelry and display are tack sharp, background velvet softens toward edges. Centered composition, jewelry occupies 55–65% of frame. High-end editorial product photography.`,
    };

    // Layer in brand-specific extra shot scenes (e.g. Taheri's signature emerald
    // walnut shot). Brand prompts win over the framework's defaults if there's
    // an id collision.
    for (const extra of brand.extraShots || []) {
        if (extra.id && extra.scenePrompt) scenes[extra.id] = extra.scenePrompt;
    }

    const scene = customPrompt ? `SCENE: ${customPrompt}` : (scenes[shotId] || scenes.ecom_hero);

    const parts = [
        base,
        anchorBlock,
        scene,
        '',
        'OUTPUT: Photorealistic — indistinguishable from a real photograph. No AI artifacts, no floating elements, no impossible reflections.',
        ...(customInstruction ? [`\nADDITIONAL DIRECTION: ${customInstruction}`] : []),
    ];

    return parts.join('\n');
}

// Build the aspect ratio / size instruction to embed in prompts (workaround for Gemini ignoring imageConfig on image-to-image)
function buildImageConfigPrompt(imageOpts) {
    const parts = [];
    if (imageOpts.aspectRatio && imageOpts.aspectRatio !== '1:1') {
        parts.push(`Generate the image in ${imageOpts.aspectRatio} aspect ratio.`);
    }
    return parts.length > 0 ? '\nIMAGE FORMAT: ' + parts.join(' ') : '';
}

// Upscale image to target resolution if the API ignores imageSize
const SIZE_PIXELS = { '512': 512, '1K': 1024, '2K': 2048, '4K': 4096 };

async function upscaleIfNeeded(base64, targetSize, aspectRatio) {
    const targetPx = SIZE_PIXELS[targetSize];
    if (!targetPx || targetPx <= 1024) return base64; // 1K or below, no upscale needed

    const buf = Buffer.from(base64, 'base64');
    const meta = await sharp(buf).metadata();
    const maxDim = Math.max(meta.width, meta.height);

    if (maxDim >= targetPx * 0.9) return base64; // already close enough

    // Calculate target dimensions preserving aspect ratio
    let w, h;
    if (meta.width >= meta.height) {
        w = targetPx;
        h = Math.round(targetPx * (meta.height / meta.width));
    } else {
        h = targetPx;
        w = Math.round(targetPx * (meta.width / meta.height));
    }

    console.log(`[Upscale] ${meta.width}x${meta.height} → ${w}x${h} (target ${targetSize})`);
    const upscaled = await sharp(buf)
        .resize(w, h, { kernel: sharp.kernel.lanczos3 })
        .png()
        .toBuffer();
    return upscaled.toString('base64');
}

const app    = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.use(express.json({ limit: '100mb' }));
// Disable caching for HTML so the user always receives the latest UI after server restarts.
// Other assets (fonts/images) can still be cached normally.
app.use(express.static(path.join(__dirname, 'public'), {
    setHeaders: (res, filePath) => {
        if (filePath.endsWith('.html')) {
            res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
            res.setHeader('Pragma', 'no-cache');
            res.setHeader('Expires', '0');
        }
    },
}));

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// ── Apply overlay to existing images ─────────────────────────────────────────
app.post('/apply-overlay', upload.array('images', 50), async (req, res) => {
    try {
        const weightText = (req.body.weightText || '').trim();
        const brandId   = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
        const files = req.files || [];
        if (files.length === 0) return res.status(400).json({ error: 'No images provided' });

        console.log(`[Overlay] brand=${brandId} applying to ${files.length} image(s) — weight: "${weightText || 'none'}"`);

        const results = [];
        for (const file of files) {
            const b64 = file.buffer.toString('base64');
            const overlaid = await applyOverlay(b64, weightText, brandId);
            const buf = Buffer.from(overlaid, 'base64');
            const meta = await sharp(buf).metadata();
            results.push({
                name: file.originalname.replace(/\.[^.]+$/, '') + '_overlay.png',
                data: overlaid,
                width: meta.width,
                height: meta.height,
            });
        }

        res.json({ success: true, results });
    } catch (err) {
        console.error('[Overlay Error]', err.message);
        res.status(500).json({ error: err.message });
    }
});

// ── Serve available providers + shot catalog to frontend ────────────────────
app.get('/providers', (_req, res) => {
    // Hide providers flagged hidden (e.g. OpenAI is fallback-only).
    const visible = {};
    for (const [k, v] of Object.entries(PROVIDERS)) if (!v.hidden) visible[k] = v;
    res.json(visible);
});
app.get('/shots', (req, res) => {
    const brandId = BRANDS[req.query.brand] ? req.query.brand : DEFAULT_BRAND;
    res.json(buildShotCatalog(brandId));
});
app.get('/brands', (_req, res) => res.json({ brands: listBrands(), defaultBrand: DEFAULT_BRAND }));

// Brand-aware shot catalog. Merges the framework's SHOT_CATALOG with any
// extraShots a brand declares (e.g. Taheri's taheri_signature emerald shot).
function buildShotCatalog(brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    const merged = { ...SHOT_CATALOG };
    for (const extra of brand.extraShots || []) {
        // The scene prompt itself is materialized inside buildShotPrompt;
        // here we only register the catalog metadata so the shot is pickable.
        const { scenePrompt, ...meta } = extra;
        merged[extra.id] = meta;
    }
    return merged;
}

// Parse overlay options off a request body. Returns null when the brand
// doesn't support overlays (so callers can short-circuit cheaply).
function parseOverlayOpts(body, brandId) {
    const brand = resolveBrand(brandId);
    const cfg = brand.overlay;
    if (!cfg || !cfg.supported) return null;
    const raw = body && body.overlayEnabled;
    let enabled;
    if (raw === undefined || raw === null || raw === '') {
        enabled = !!cfg.defaultEnabled;
    } else {
        enabled = raw === true || raw === '1' || raw === 1 || raw === 'true';
    }
    const weightText = (body && typeof body.weightText === 'string') ? body.weightText : '';
    return { enabled, weightText };
}
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

// ── Model capabilities endpoint ────────────────────────────────────────────
app.get('/model-capabilities', (_req, res) => {
    res.json({
        gemini: {
            model: 'gemini-3-pro-image-preview',
            label: 'Gemini 3 Pro',
            aspectRatios: ['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'],
            imageSizes: ['1K', '2K', '4K'],
            defaultAspectRatio: '1:1',
            defaultImageSize: '1K',
        },
        nanobana2: {
            model: 'gemini-3.1-flash-image-preview',
            label: 'Gemini 3.1 Flash',
            aspectRatios: ['1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1', '4:3', '4:5', '5:4', '8:1', '9:16', '16:9', '21:9'],
            imageSizes: ['512', '1K', '2K', '4K'],
            defaultAspectRatio: '1:1',
            defaultImageSize: '2K',
        },
        openai: {
            model: 'gpt-image-1.5',
            label: 'GPT Image 1.5',
            aspectRatios: ['1:1'],
            imageSizes: ['1024x1024'],
            defaultAspectRatio: '1:1',
            defaultImageSize: '1024x1024',
        },
    });
});

// ── Single product generation ───────────────────────────────────────────────
app.post('/generate', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const shotIds           = JSON.parse(req.body.shots || '[]');
    const customInstruction = (req.body.customInstruction || '').trim() || null;
    const customPrompt      = (req.body.customPrompt || '').trim() || null;
    const provider          = (req.body.provider || 'gemini').trim();
    const aspectRatios      = JSON.parse(req.body.aspectRatios || '["1:1"]');
    const imageSize         = (req.body.imageSize || '').trim() || null;
    const variationCount    = Math.min(Math.max(parseInt(req.body.variationCount) || 1, 1), 5);
    const overlayEnabled    = req.body.overlayEnabled === 'true';
    const weightText        = (req.body.weightText || '').trim();
    const overlayOpts       = { enabled: overlayEnabled, weightText };
    const brandId           = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
    const shotCatalog       = buildShotCatalog(brandId);

    if (shotIds.length === 0) return res.status(400).json({ error: 'No shots selected.' });

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        const totalImages = shotIds.length * aspectRatios.length * variationCount;
        console.log(`[Generate] brand=${brandId} ${shotIds.length} shot(s) × ${aspectRatios.length} ratio(s) × ${variationCount} var(s) = ${totalImages} images via ${provider}${overlayEnabled ? ` [overlay: ${weightText || 'logo only'}]` : ''}`);

        // ── Anchor-first consistency pipeline ──
        const results = [];
        let anchorRef = null;

        const anchorId = shotIds.includes('ecom_hero') ? 'ecom_hero'
            : shotIds.find(id => id.startsWith('ecom_'))
            || shotIds[0];

        // Generate anchor shot once (first aspect ratio, variation 1)
        console.log(`[Anchor] Generating ${anchorId} as consistency anchor via ${provider}...`);
        const anchorShot = shotCatalog[anchorId];
        const anchorAR = aspectRatios[0];
        const anchorData = await generateShot(anchorId, imageInputs, customInstruction, false, provider, { aspectRatio: anchorAR, imageSize }, overlayOpts, customPrompt, brandId);
        // Store clean (no-overlay) version as anchor reference for consistency
        const anchorClean = await generateWithGemini ? anchorData : anchorData; // already generated
        anchorRef = { base64: anchorData, mimeType: 'image/png' };
        results.push({ id: anchorId, label: anchorShot.label, category: anchorShot.category, data: anchorData, aspectRatio: anchorAR, variation: 1 });

        // Build all remaining tasks: (shot × ratio × variation) minus the anchor we already did
        const tasks = [];
        for (const shotId of shotIds) {
            const shot = shotCatalog[shotId];
            if (!shot) continue;
            for (const ar of aspectRatios) {
                for (let v = 1; v <= variationCount; v++) {
                    // Skip the anchor we already generated
                    if (shotId === anchorId && ar === anchorAR && v === 1) continue;
                    tasks.push({ shotId, shot, ar, v, isAnchorShot: shotId === anchorId });
                }
            }
        }

        if (tasks.length > 0) {
            const refsWithAnchor = [...imageInputs, anchorRef];
            const parallel = await Promise.all(tasks.map(async ({ shotId, shot, ar, v, isAnchorShot }) => {
                const refs = isAnchorShot ? imageInputs : refsWithAnchor;
                const hasAnchor = !isAnchorShot;
                const data = await generateShot(shotId, refs, customInstruction, hasAnchor, provider, { aspectRatio: ar, imageSize }, overlayOpts, customPrompt, brandId);
                return { id: shotId, label: shot.label, category: shot.category, data, aspectRatio: ar, variation: v };
            }));
            results.push(...parallel.filter(Boolean));
        }

        // Sort: by shot order, then aspect ratio order, then variation
        const ordered = [];
        for (const shotId of shotIds) {
            for (const ar of aspectRatios) {
                for (let v = 1; v <= variationCount; v++) {
                    const match = results.find(r => r.id === shotId && r.aspectRatio === ar && r.variation === v);
                    if (match) ordered.push(match);
                }
            }
        }

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
    const customPrompt      = (req.body.customPrompt || '').trim() || null;
    const provider          = (req.body.provider || 'gemini').trim();
    const aspectRatio       = (req.body.aspectRatio || '').trim() || null;
    const imageSize         = (req.body.imageSize || '').trim() || null;
    const overlayEnabled    = req.body.overlayEnabled === 'true';
    const weightText        = (req.body.weightText || '').trim();
    const overlayOpts       = { enabled: overlayEnabled, weightText };
    const brandId           = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        const shot = buildShotCatalog(brandId)[shotId];
        if (!shot) return res.status(400).json({ error: 'Unknown shot type.' });

        const imageData = await generateShot(shotId, imageInputs, customInstruction, false, provider, { aspectRatio, imageSize }, overlayOpts, customPrompt, brandId);
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
    const customPrompt      = (req.query.customPrompt || '').trim() || null;
    const shotIds           = JSON.parse(req.query.shots || '[]');
    const provider          = (req.query.provider || 'gemini').trim();
    const aspectRatios      = JSON.parse(req.query.aspectRatios || '["1:1"]');
    const imageSize         = (req.query.imageSize || '').trim() || null;
    const variationCount    = Math.min(Math.max(parseInt(req.query.variationCount) || 1, 1), 5);
    const brandId           = BRANDS[req.query.brand] ? req.query.brand : DEFAULT_BRAND;
    const shotCatalog       = buildShotCatalog(brandId);

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
            const anchorShot = shotCatalog[anchorId];
            const anchorAR = aspectRatios[0];

            send({ type: 'angle_start', product: productName, angle: anchorId, label: `${anchorShot.label} (anchor)` });
            try {
                const b64 = await generateShot(anchorId, imageInputs, customInstruction, false, provider, { aspectRatio: anchorAR, imageSize }, {}, customPrompt, brandId);
                anchorRef = { base64: b64, mimeType: 'image/png' };
                const outPath = path.join(outDir, `${anchorId}_${anchorAR.replace(':', 'x')}_v1.png`);
                fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
                send({ type: 'angle_done', product: productName, angle: anchorId, label: `${anchorShot.label} · ${anchorAR}`, savedTo: outPath });
                send({ type: 'usage', usage: usageStats });
            } catch (err) {
                send({ type: 'angle_error', product: productName, angle: anchorId, message: err.message });
            }

            if (batchCancelled) {
                send({ type: 'product_done', product: productName });
                send({ type: 'cancelled', message: 'Batch cancelled by user.' });
                break;
            }

            // Build all remaining tasks
            const batchTasks = [];
            for (const shotId of shotIds) {
                const shot = shotCatalog[shotId];
                if (!shot) continue;
                for (const ar of aspectRatios) {
                    for (let v = 1; v <= variationCount; v++) {
                        if (shotId === anchorId && ar === anchorAR && v === 1) continue;
                        batchTasks.push({ shotId, shot, ar, v, isAnchorShot: shotId === anchorId });
                    }
                }
            }

            const refsWithAnchor = anchorRef ? [...imageInputs, anchorRef] : imageInputs;

            for (const t of batchTasks) {
                send({ type: 'angle_start', product: productName, angle: t.shotId, label: `${t.shot.label} · ${t.ar}${t.v > 1 ? ` #${t.v}` : ''}` });
            }

            const parallelTasks = batchTasks.map(({ shotId, shot, ar, v, isAnchorShot }) => {
                const refs = isAnchorShot ? imageInputs : refsWithAnchor;
                const hasAnchor = !isAnchorShot;
                return generateShot(shotId, refs, customInstruction, hasAnchor, provider, { aspectRatio: ar, imageSize }, {}, customPrompt, brandId)
                    .then(b64 => {
                        const suffix = variationCount > 1 ? `_v${v}` : '';
                        const p = path.join(outDir, `${shotId}_${ar.replace(':', 'x')}${suffix}.png`);
                        fs.writeFileSync(p, Buffer.from(b64, 'base64'));
                        send({ type: 'angle_done', product: productName, angle: shotId, label: `${shot.label} · ${ar}${v > 1 ? ` #${v}` : ''}`, savedTo: p });
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
    const { productFolder, angleId, provider: retryProvider, aspectRatio: retryAR, imageSize: retryIS, brand: rawBrand } = req.body;
    const provider = (retryProvider || 'gemini').trim();
    const aspectRatio = (retryAR || '').trim() || null;
    const imageSize = (retryIS || '').trim() || null;
    const brandId = BRANDS[rawBrand] ? rawBrand : DEFAULT_BRAND;
    if (!productFolder || !angleId) return res.status(400).json({ error: 'Missing productFolder or angleId.' });

    const shot = buildShotCatalog(brandId)[angleId];
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

        const raw = await generateShot(angleId, imageInputs, null, false, provider, { aspectRatio, imageSize }, {}, null, brandId);
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
    const extraContext  = (req.body.extraContext || '').trim();
    const brandId       = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
    const brand         = resolveBrand(brandId);

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

    const captionPrompt = `${brand.captionSystem}

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
"Meet your new obsession. *A certified yellow sapphire. Brilliant zircon accents. 925 sterling silver*. A combination this stunning doesn't come along often — and at ${brand.captionBrandMention}, it's entirely yours. We're celebrating our soft launch with special introductory pricing. These pieces won't wait forever. 📩 DM to order 🇵🇰 Nationwide Delivery"

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
                const response = await withTimeout(
                    geminiClient.models.generateContent({
                        model: 'gemini-3-flash-preview',
                        contents: [{ role: 'user', parts }],
                    }),
                    30000,
                    'caption'
                );
                const resParts = response.candidates?.[0]?.content?.parts || [];
                captionText = resParts.map(p => p.text).filter(Boolean).join('').trim();
            } finally {
                releaseGeminiSlot();
            }
        } else if (geminiClient) {
            // Text-only (no images)
            await acquireGeminiSlot();
            try {
                const response = await withTimeout(
                    geminiClient.models.generateContent({
                        model: 'gemini-3-flash-preview',
                        contents: [{ role: 'user', parts: [{ text: captionPrompt }] }],
                    }),
                    20000,
                    'caption'
                );
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

// ── AI prompt naming endpoint ───────────────────────────────────────────────
// Takes a prompt body, returns a short editorial label suitable as the saved-prompt name.
app.post('/name-prompt', async (req, res) => {
    const body = (req.body?.body || '').toString().trim();
    if (!body) return res.status(400).json({ error: 'Prompt body is required.' });
    if (!geminiClient) return res.status(500).json({ error: 'No AI provider available.' });

    const namingPrompt = `You are naming a saved prompt in a luxury jewelry photography pipeline. The user has written a creative direction / photographic prompt, and you must produce ONE short editorial label for it — the kind of name you'd see in a moodboard or shot list.

REQUIREMENTS:
- 2 to 5 words maximum
- Title Case (capitalise each significant word)
- No quotes, no emojis, no punctuation (no periods, commas, dashes, ellipsis)
- Evocative and specific to THIS prompt — capture the mood, setting, lighting, or core visual idea
- Never generic ("Nice Shot", "Jewelry Photo", "Beautiful Image")
- Never copy verbatim phrases from the prompt — distill the essence

EXAMPLES OF GOOD NAMES:
- "Golden Hour Intimacy"
- "Velvet Atelier"
- "Cinematic Close Crop"
- "Marble Pedestal Study"
- "Soft Dawn Portrait"

THE PROMPT TO NAME:
"""
${body}
"""

Output ONLY the name — nothing else. No explanation. No quotes. Just the words.`;

    const t0 = Date.now();
    console.log('[Name Prompt] Start, body length:', body.length);
    try {
        await acquireGeminiSlot();
        let nameText;
        try {
            const response = await withTimeout(
                geminiClient.models.generateContent({
                    model: 'gemini-3-flash-preview',
                    contents: [{ role: 'user', parts: [{ text: namingPrompt }] }],
                    config: { thinkingConfig: { thinkingLevel: 'MINIMAL' } },
                }),
                10000,
                'name-prompt'
            );
            const parts = response.candidates?.[0]?.content?.parts || [];
            nameText = parts.map(p => p.text).filter(Boolean).join('').trim();
        } finally {
            releaseGeminiSlot();
        }

        // Strip quotes, markdown, trailing punctuation, and cap the length
        nameText = (nameText || '')
            .replace(/^["'`*_]+|["'`*_.,!?…\-]+$/g, '')
            .replace(/^```[\s\S]*?\n/, '')
            .replace(/\n```$/, '')
            .replace(/\s+/g, ' ')
            .trim();

        if (!nameText) {
            console.warn('[Name Prompt] Empty result after', Date.now() - t0, 'ms');
            return res.status(500).json({ error: 'AI returned an empty name.' });
        }
        if (nameText.length > 60) nameText = nameText.slice(0, 57).trimEnd() + '…';

        console.log('[Name Prompt] OK in', Date.now() - t0, 'ms ->', nameText);
        res.json({ success: true, name: nameText });
    } catch (err) {
        console.error('[Name Prompt Error] after', Date.now() - t0, 'ms:', err?.message || err);
        res.status(500).json({ error: err.message || 'Name generation failed.' });
    }
});

// ── WhatsApp community post endpoint ────────────────────────────────────────
// Takes a generated jewelry image (base64) and optional weight string.
// Returns a copy-paste-ready WhatsApp post following the Taheri format.
app.post('/whatsapp-post', async (req, res) => {
    const imageB64 = (req.body?.imageB64 || '').toString();
    const mimeType = (req.body?.mimeType || 'image/png').toString();
    const weight = (req.body?.weight || '').toString().trim();

    if (!imageB64) return res.status(400).json({ error: 'Image data is required.' });
    if (!geminiClient) return res.status(500).json({ error: 'No AI provider available.' });

    const userSpecsBlock = weight
        ? `\n\nUSER-PROVIDED SPECS (these override visual guesses):\n- Weight: ${weight}`
        : '';

    const postPrompt = `You are a high-end jewelry branding expert writing a WhatsApp community post for Taheri (taheri.shop). Analyze the attached jewelry image and produce ONE elegant, copy-paste-ready WhatsApp post.

ABSOLUTE RULES:
- Output ONLY the final formatted post. No greetings, no explanations, no preamble, no sign-off.
- Response must be 100% ready to copy and paste — no surrounding code fences, no quotes around it.
- Never use generic names like "gold ring" or "diamond bracelet" — invent a unique poetic name.

STEP 1 — Analyze the image:
- Item type: ring, bangle, bracelet, earring, pendant, necklace, set, etc.
- Metal: Yellow / Rose / White gold, with purity if visible (18K, 21K, 22K, 24K). If unclear, write the colour only without a purity figure.
- Stones: ruby, emerald, zircon, pearl, diamond, CZ — or "No Stones / Pure Gold" if none.
- Design vibe: minimalist, ornate, vintage, geometric, floral, Arabic-inspired, bridal, statement, everyday.
- Occasion suitability: wedding, casual, formal, festive, gifting, everyday.

STEP 2 — Invent a poetic name. Examples of the right register:
- The Celestial Arc (curved bangle)
- The Gilded Reverie (delicate gold ring)
- The Ember Bloom (ruby floral pendant)
- The Quiet Storm (bold geometric bracelet)
Never plain descriptive names. Evoke emotion or imagery.

STEP 3 — Write the post in EXACTLY this format (WhatsApp markdown, asterisks for bold, underscores for italic):

✨ *[Unique Poetic Name] [Item Type]* — _[Metal & Purity] | [Weight]_

[One hook sentence — elegant, punchy, evocative. Captures the design vibe in one breath.]

✦ *Stone:* [Specific stone(s) OR "No Stones / Pure Gold"]
✦ *Where to wear:* [Short occasion detail — 5 to 10 words]
✦ *Style:* [Short aesthetic detail — 5 to 10 words]

*Shop Now:*
💬 WhatsApp: +923352275553, +923262275554
📍 Visit: Najmi Market, Shop #40 & #16
🌐 Browse all designs at *taheri.shop*

FORMATTING RULES:
- Header line: name+type in *bold*, metal/weight in _italic_.
- Bullet labels (Stone, Where to wear, Style) in *bold*.
- Do NOT add extra bullets or sections.
- Do NOT repeat metal or weight inside the bullets — they live in the header.
- Bullet values: short punchy phrases, not full sentences.
- Hook sentence: complete and standalone, poetic but not overwrought.

EDGE CASES:
- If weight is not provided, write "Weight on request" in the header.
- If the image shows multiple items as a set, name it as a set (e.g. *The Dusk Duet Set*) and list all stones together.
- If purity is unclear, write the metal colour only (e.g. "Yellow Gold").${userSpecsBlock}

Output ONLY the formatted post. Begin with the ✨ line.`;

    const t0 = Date.now();
    console.log('[WhatsApp Post] Start, weight:', weight || '(none)');

    try {
        await acquireGeminiSlot();
        let postText;
        try {
            const response = await withTimeout(
                geminiClient.models.generateContent({
                    model: 'gemini-3-flash-preview',
                    contents: [{
                        role: 'user',
                        parts: [
                            { text: postPrompt },
                            { inlineData: { mimeType, data: imageB64 } },
                        ],
                    }],
                }),
                30000,
                'whatsapp-post'
            );
            const parts = response.candidates?.[0]?.content?.parts || [];
            postText = parts.map(p => p.text).filter(Boolean).join('').trim();
        } finally {
            releaseGeminiSlot();
        }

        // Strip wrapping code fences and stray quotes if the model added them
        postText = (postText || '')
            .replace(/^```[a-z]*\s*\n?/i, '')
            .replace(/\n?```\s*$/, '')
            .trim();

        if (!postText) {
            console.warn('[WhatsApp Post] Empty result after', Date.now() - t0, 'ms');
            return res.status(500).json({ error: 'AI returned an empty post.' });
        }

        console.log('[WhatsApp Post] OK in', Date.now() - t0, 'ms, length:', postText.length);
        res.json({ success: true, post: postText });
    } catch (err) {
        console.error('[WhatsApp Post Error] after', Date.now() - t0, 'ms:', err?.message || err);
        res.status(500).json({ error: err.message || 'WhatsApp post generation failed.' });
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
    res.setHeader('Content-Disposition', 'attachment; filename="taheri-shots.zip"');
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
async function generateShot(shotId, imageInputs, customInstruction, hasAnchor = false, provider = 'gemini', imageOpts = {}, overlayOpts = {}, customPrompt = null, brandId = DEFAULT_BRAND) {
    const prompt = buildShotPrompt(shotId, customInstruction, hasAnchor, customPrompt, brandId) + buildImageConfigPrompt(imageOpts);

    let result;
    if (provider === 'openai') {
        result = await generateWithOpenAI(prompt, imageInputs, imageOpts);
    } else if (provider === 'nanobana2') {
        result = await generateWithNanoBana2(prompt, imageInputs, imageOpts);
    } else {
        result = await generateWithGemini(prompt, imageInputs, imageOpts);
    }

    trackUsage(provider, shotId, imageOpts.imageSize);

    // Apply overlay if enabled — brand-aware logo + weight composite.
    if (overlayOpts.enabled) {
        result = await applyOverlay(result, overlayOpts.weightText || '', brandId);
    }

    // Log final dimensions
    const finalBuf = Buffer.from(result, 'base64');
    const finalMeta = await sharp(finalBuf).metadata();
    console.log(`[Final] ${shotId}: ${finalMeta.width}x${finalMeta.height} (${(finalBuf.length / 1024 / 1024).toFixed(1)}MB)`);

    return result;
}

async function generateWithGemini(prompt, imageInputs, imageOpts = {}) {
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    let raw = await callGemini(parts, 0, imageOpts);
    // Only force square if no aspect ratio specified
    if (!imageOpts.aspectRatio || imageOpts.aspectRatio === '1:1') {
        raw = await makeSquareBase64(raw);
    }
    // Upscale if API ignored imageSize
    if (imageOpts.imageSize) raw = await upscaleIfNeeded(raw, imageOpts.imageSize, imageOpts.aspectRatio);
    return raw;
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

async function generateWithNanoBana2(prompt, imageInputs, imageOpts = {}) {
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    let raw = await callNanoBana2(parts, 0, imageOpts);
    if (!imageOpts.aspectRatio || imageOpts.aspectRatio === '1:1') {
        raw = await makeSquareBase64(raw);
    }
    // Upscale if API ignored imageSize
    if (imageOpts.imageSize) raw = await upscaleIfNeeded(raw, imageOpts.imageSize, imageOpts.aspectRatio);
    return raw;
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

// Race a promise against a timeout so a hung SDK call surfaces as an error
// instead of blocking a slot indefinitely.
function withTimeout(promise, ms, label = 'gemini') {
    return new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
        promise.then(
            v => { clearTimeout(t); resolve(v); },
            e => { clearTimeout(t); reject(e); }
        );
    });
}

async function callGemini(parts, attempt = 0, imageOpts = {}) {
    await acquireGeminiSlot();
    try {
        const imgConfig = {};
        if (imageOpts.aspectRatio) imgConfig.aspectRatio = imageOpts.aspectRatio;
        if (imageOpts.imageSize) imgConfig.imageSize = imageOpts.imageSize;

        console.log(`[Gemini] calling... (${parts.filter(p => p.inlineData).length} image(s))${attempt > 0 ? ` [retry ${attempt}]` : ''}${Object.keys(imgConfig).length ? ` [${JSON.stringify(imgConfig)}]` : ''}`);
        const config = { responseModalities: ['TEXT', 'IMAGE'] };
        if (Object.keys(imgConfig).length > 0) config.imageConfig = imgConfig;

        const response = await geminiClient.models.generateContent({
            model: 'gemini-3-pro-image-preview',
            contents: [{ role: 'user', parts }],
            config,
        });

        const resParts  = response.candidates?.[0]?.content?.parts || [];
        const imagePart = resParts.find(p => p.inlineData?.data && !p.thought);
        if (!imagePart) {
            const text = resParts.find(p => p.text)?.text || 'none';
            console.error('[Gemini] No image. Response text:', text.slice(0, 300));
            throw new Error('Gemini returned no image — ' + text.slice(0, 120));
        }

        const buf = Buffer.from(imagePart.inlineData.data, 'base64');
        const meta = await sharp(buf).metadata();
        if (!meta.width || !meta.height) throw new Error('Gemini returned invalid image data');

        console.log(`[Gemini] image OK — native ${meta.width}x${meta.height} (${(buf.length / 1024 / 1024).toFixed(1)}MB)${imgConfig.imageSize ? ` [requested ${imgConfig.imageSize}]` : ''}`);
        return imagePart.inlineData.data;
    } catch (err) {
        if (attempt < MAX_RETRIES - 1) {
            const delay = RETRY_DELAYS[attempt] || 5000;
            console.log(`[Gemini] retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms...`);
            await new Promise(r => setTimeout(r, delay));
            return callGemini(parts, attempt + 1, imageOpts);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

async function callNanoBana2(parts, attempt = 0, imageOpts = {}) {
    await acquireGeminiSlot();
    try {
        const imgConfig = {
            aspectRatio: imageOpts.aspectRatio || '1:1',
            imageSize: imageOpts.imageSize || '2K',
        };
        console.log(`[NanoBana2] calling gemini-3.1-flash-image-preview... (${parts.filter(p => p.inlineData).length} image(s))${attempt > 0 ? ` [retry ${attempt}]` : ''} [${JSON.stringify(imgConfig)}]`);
        const response = await geminiClient.models.generateContent({
            model: 'gemini-3.1-flash-image-preview',
            contents: [{ role: 'user', parts }],
            config: {
                responseModalities: ['TEXT', 'IMAGE'],
                imageConfig: imgConfig,
            },
        });

        const resParts  = response.candidates?.[0]?.content?.parts || [];
        const imagePart = resParts.find(p => p.inlineData?.data && !p.thought);
        if (!imagePart) {
            const text = resParts.find(p => p.text)?.text || 'none';
            console.error('[NanoBana2] No image. Response text:', text.slice(0, 300));
            throw new Error('Nano Banana 2 returned no image — ' + text.slice(0, 120));
        }

        const buf = Buffer.from(imagePart.inlineData.data, 'base64');
        const meta = await sharp(buf).metadata();
        if (!meta.width || !meta.height) throw new Error('Nano Banana 2 returned invalid image data');

        console.log(`[NanoBana2] image OK — native ${meta.width}x${meta.height} (${(buf.length / 1024 / 1024).toFixed(1)}MB) [requested ${imgConfig.imageSize}]`);
        return imagePart.inlineData.data;
    } catch (err) {
        if (attempt < MAX_RETRIES - 1) {
            const delay = RETRY_DELAYS[attempt] || 5000;
            console.log(`[NanoBana2] retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms...`);
            await new Promise(r => setTimeout(r, delay));
            return callNanoBana2(parts, attempt + 1, imageOpts);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

// ── Overlay: weight text + logo ────────────────────────────────────────────
const LOGO_PATH = path.join(__dirname, 'public', 'assets', 'taheri-light.png');
// Futura LT Light must be installed in the system/user fonts directory for librsvg to find it
const FUTURA_FONT_FAMILY = 'Futura LT';

async function applyOverlay(base64, weightText, brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    const overlayCfg = brand.overlay;
    if (!overlayCfg || !overlayCfg.supported) return base64;

    const buf = Buffer.from(base64, 'base64');
    const meta = await sharp(buf).metadata();
    const w = meta.width;
    const h = meta.height;

    // Scale overlay relative to image size (reference: 3000x3000px canvas)
    const pad = Math.round(w * (125 / 3000));
    const fontSize = Math.round(w * (143 / 3000));
    const logoWidth = Math.round(w * (580 / 3000));

    const composites = [];

    // Weight text (top-left) — SVG, embedded Jost Light (Futura alternative)
    if (weightText && weightText.trim()) {
        const textLeftPad = Math.round(w * (120 / 3000));
        const textSvg = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${fontSize * 2}">
            <text x="${textLeftPad}" y="${fontSize * 1.1}" font-family="${FUTURA_FONT_FAMILY}" font-size="${fontSize}" font-weight="300" fill="white" letter-spacing="2">${weightText.trim()}</text>
        </svg>`);
        const textPad = Math.round(w * (100 / 3000));
        composites.push({ input: textSvg, top: textPad, left: 0, });
    }

    // Brand logo (top-right) — path comes from brands.js so each brand can ship
    // its own asset variant.
    const logoAbsPath = overlayCfg.logoPath && path.isAbsolute(overlayCfg.logoPath)
        ? overlayCfg.logoPath
        : path.join(__dirname, overlayCfg.logoPath || '');
    if (logoAbsPath && fs.existsSync(logoAbsPath)) {
        const logoBuf = await sharp(logoAbsPath)
            .resize({ width: logoWidth, fit: 'inside' })
            .png()
            .toBuffer();
        const logoMeta = await sharp(logoBuf).metadata();
        composites.push({
            input: logoBuf,
            top: pad,
            left: w - logoMeta.width - pad,
        });
    }

    if (composites.length === 0) return base64;

    console.log(`[Overlay] brand=${brand.id} ${weightText ? 'weight "' + weightText.trim() + '"' : 'no weight'} + logo on ${w}x${h}`);
    const result = await sharp(buf)
        .composite(composites)
        .png({ compressionLevel: 6 })
        .toBuffer();
    return result.toString('base64');
}

// ── Image helpers ───────────────────────────────────────────────────────────
async function makeSquareBase64(base64) {
    const buf = Buffer.from(base64, 'base64');
    const meta = await sharp(buf).metadata();
    // If already square (or very close), return as-is
    if (Math.abs(meta.width - meta.height) <= 2) return base64;
    // Crop to square from center instead of padding with white
    const size = Math.min(meta.width, meta.height);
    const out = await sharp(buf)
        .extract({
            left: Math.floor((meta.width - size) / 2),
            top: Math.floor((meta.height - size) / 2),
            width: size,
            height: size,
        })
        .png()
        .toBuffer();
    return out.toString('base64');
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
app.listen(PORT, () => console.log(`\nTaheri Pipeline → http://localhost:${PORT}\n`));
