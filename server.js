require('dotenv').config();

const express      = require('express');
const multer       = require('multer');
const path         = require('path');
const fs           = require('fs');
const os           = require('os');
const crypto       = require('crypto');
const { execFile } = require('child_process');
const sharp        = require('sharp');
const { GoogleGenAI } = require('@google/genai');
const OpenAI       = require('openai');
const { BRANDS, DEFAULT_BRAND, resolveBrand, listBrands } = require('./brands');

process.on('uncaughtException', (err) => {
    console.error('[Uncaught Exception]', err.message, err.stack);
});
process.on('unhandledRejection', (reason) => {
    console.error('[Unhandled Rejection]', reason);
});

const PORT = process.env.PORT || 3000;

// ── Provider clients ───────────────────────────────────────────────────────
// Auto-detect Vertex AI Express Mode keys (AQ.*) vs AI Studio keys (AIza*)
const googleKey = process.env.GOOGLE_API_KEY;
const isVertexKey = googleKey && googleKey.startsWith('AQ.');
const geminiClient = googleKey
    ? (isVertexKey
        ? new GoogleGenAI({ vertexai: true, apiKey: googleKey })
        : new GoogleGenAI({ apiKey: googleKey }))
    : null;
if (googleKey) console.log(`[Gemini] Using ${isVertexKey ? 'Vertex AI Express Mode' : 'AI Studio'} key`);
const openaiClient = process.env.OPENAI_API_KEY  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

// Which providers are available
const PROVIDERS = {};
if (geminiClient) PROVIDERS.gemini    = { label: 'Gemini',          canGenerate: true };
if (openaiClient) PROVIDERS.openai    = { label: 'OpenAI',          canGenerate: true };
if (geminiClient) PROVIDERS.nanobana2 = { label: 'Nano Banana 2',   canGenerate: true };

console.log('[Providers]', Object.keys(PROVIDERS).join(', ') || 'NONE — add API keys to .env');

// ── Cost tracking ─────────────────────────────────────────────────────────
const COST_PER_IMAGE_BASE = {
    gemini:    0.134,   // Nano Banana Pro @ 1K
    nanobana2: 0.101,   // Nano Banana 2 @ 2K
    openai:    0.133,   // GPT Image 1.5 High @ 1024x1024
};
// Resolution-aware cost: draft costs ~50% less
function getCostPerImage(provider, resolution = 'standard') {
    const base = COST_PER_IMAGE_BASE[provider] || 0;
    if (resolution === 'draft') return base * 0.5;
    return base;
}
// Keep COST_PER_IMAGE as a static reference for backward compat (standard resolution)
const COST_PER_IMAGE = COST_PER_IMAGE_BASE;

const usageStats = {
    session: {
        gemini:    { images: 0, cost: 0 },
        nanobana2: { images: 0, cost: 0 },
        openai:    { images: 0, cost: 0 },
        total:     { images: 0, cost: 0 },
    },
    history: [],  // last 50 entries
};

function trackUsage(provider, shotId, resolution = 'standard') {
    const cost = getCostPerImage(provider, resolution);
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
    debounceSaveUsageStats();
    return entry;
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Pipeline enhancements: caches, QC, fallback, budget, audit, post-process
// ═══════════════════════════════════════════════════════════════════════════

const CACHE_DIR        = path.join(__dirname, '.cache');
const OUTPUT_CACHE_DIR = path.join(CACHE_DIR, 'outputs');
const JPEG_CACHE_DIR   = path.join(CACHE_DIR, 'jpeg');
const AUDIT_LOG_PATH   = path.join(__dirname, 'audit.log');
const USAGE_STATS_PATH = path.join(CACHE_DIR, 'usage-stats.json');
fs.mkdirSync(OUTPUT_CACHE_DIR, { recursive: true });
fs.mkdirSync(JPEG_CACHE_DIR,   { recursive: true });

function sha1(buf) {
    return crypto.createHash('sha1').update(buf).digest('hex');
}

// ── Output cache (skips API call if same prompt+refs were generated before) ──
function outputCacheKey(provider, shotId, prompt, imageInputs) {
    const refHashes = imageInputs
        .map(i => sha1(Buffer.from(i.base64, 'base64')))
        .sort()
        .join(',');
    return sha1(`${provider}|${shotId}|${prompt}|${refHashes}`);
}
function outputCacheGet(key) {
    const p = path.join(OUTPUT_CACHE_DIR, `${key}.png`);
    return fs.existsSync(p) ? fs.readFileSync(p).toString('base64') : null;
}
function outputCachePut(key, base64) {
    try {
        fs.writeFileSync(path.join(OUTPUT_CACHE_DIR, `${key}.png`), Buffer.from(base64, 'base64'));
    } catch (e) { /* non-fatal */ }
}

// ── JPEG cache (skips sips/sharp HEIC conversion on re-runs) ──
function jpegCacheGet(hash) {
    const p = path.join(JPEG_CACHE_DIR, `${hash}.jpg`);
    return fs.existsSync(p) ? fs.readFileSync(p) : null;
}
function jpegCachePut(hash, jpegBuffer) {
    try {
        fs.writeFileSync(path.join(JPEG_CACHE_DIR, `${hash}.jpg`), jpegBuffer);
    } catch (e) { /* non-fatal */ }
}

// ── Audit log (one JSONL line per generation) ──
function audit(entry) {
    try {
        fs.appendFileSync(AUDIT_LOG_PATH, JSON.stringify({ ts: Date.now(), ...entry }) + '\n');
    } catch (e) { /* non-fatal */ }
}

// ── Budget cap (hard stop if session cost exceeds $DAILY_BUDGET_USD) ──
const DAILY_BUDGET_USD = parseFloat(process.env.DAILY_BUDGET_USD || '0') || null;
function budgetCheck() {
    if (!DAILY_BUDGET_USD) return;
    if (usageStats.session.total.cost >= DAILY_BUDGET_USD) {
        throw new Error(
            `Budget cap reached: session cost $${usageStats.session.total.cost.toFixed(3)} ≥ ` +
            `$${DAILY_BUDGET_USD.toFixed(2)}. Raise DAILY_BUDGET_USD in .env or restart server to reset.`
        );
    }
}

// ── Post-process ecom shots: white-point lock + gentle sharpen ──
async function postProcessEcom(base64) {
    try {
        const input = Buffer.from(base64, 'base64');
        const img   = sharp(input);
        const meta  = await img.metadata();
        if (!meta.width || !meta.height) return base64;

        // Sample four corners to see if the background is meant to be white
        const corners = await Promise.all([
            img.clone().extract({ left: 0, top: 0, width: 8, height: 8 }).stats(),
            img.clone().extract({ left: meta.width - 8, top: 0, width: 8, height: 8 }).stats(),
            img.clone().extract({ left: 0, top: meta.height - 8, width: 8, height: 8 }).stats(),
            img.clone().extract({ left: meta.width - 8, top: meta.height - 8, width: 8, height: 8 }).stats(),
        ]);
        const allNearWhite = corners.every(s =>
            s.channels.slice(0, 3).every(c => c.mean >= 240)
        );

        let pipeline = sharp(input).sharpen({ sigma: 0.6 });
        if (allNearWhite) {
            // Clamp the near-white background to pure #FFFFFF using a tight levels curve
            pipeline = pipeline.linear(1.06, -12); // slight contrast lift
        }
        const out = await pipeline.png({ compressionLevel: 6 }).toBuffer();
        return out.toString('base64');
    } catch (e) {
        console.warn('[PostProcess] skipped:', e.message);
        return base64;
    }
}

// ── QC pass: cheap Gemini text call to score fidelity vs references ──
const QC_ENABLED   = process.env.QC_ENABLED !== '0';
const QC_THRESHOLD = parseFloat(process.env.QC_THRESHOLD || '6.5');
const QC_MODEL     = 'gemini-3.1-pro-preview';

async function qcShot(generatedBase64, imageInputs, shotLabel) {
    if (!QC_ENABLED || !geminiClient) return { score: 10, defects: '', skipped: true };
    try {
        const parts = [
            { text:
`You are a QA inspector for jewelry product photography. You will receive:
  1. A GENERATED shot (first image) of type: ${shotLabel}
  2. One or more REFERENCE photos of the actual jewelry piece

Score the generated shot 0–10 on FIDELITY to the reference (is it the same piece?):
- Stone count, stone color, stone cut
- Metal tone (yellow gold / rose gold / silver)
- Proportions and overall design
- Setting style, prong count, pavé pattern
Reply in EXACTLY this format, two lines only:
SCORE: <number 0–10>
DEFECTS: <one short sentence listing any mismatches, or "none">` },
            { inlineData: { mimeType: 'image/png',  data: generatedBase64 } },
            ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
        ];
        const response = await withTimeout(
            geminiClient.models.generateContent({
                model: QC_MODEL,
                contents: [{ role: 'user', parts }],
            }),
            30000,
            'QC'
        );
        const text = (response.candidates?.[0]?.content?.parts || [])
            .map(p => p.text).filter(Boolean).join('').trim();
        const scoreMatch = text.match(/SCORE:\s*(\d+(?:\.\d+)?)/i);
        const defMatch   = text.match(/DEFECTS:\s*(.+)/i);
        const score   = scoreMatch ? parseFloat(scoreMatch[1]) : 10;
        const defects = defMatch   ? defMatch[1].trim()        : '';
        return { score, defects, skipped: false };
    } catch (e) {
        console.warn('[QC] skipped:', e.message);
        return { score: 10, defects: '', skipped: true };
    }
}

// ── Provider fallback chain (used when primary 429s out) ──
const PROVIDER_FALLBACK = {
    gemini:    ['openai', 'nanobana2'],
    nanobana2: ['gemini', 'openai'],
    openai:    ['gemini', 'nanobana2'],
};
function availableFallbacks(primary) {
    return (PROVIDER_FALLBACK[primary] || []).filter(p => PROVIDERS[p]);
}
function isFallbackWorthy(err) {
    // 429/quota or hard-capped provider errors → fall back. Prompt/safety errors → don't.
    const msg = String(err?.message || '').toLowerCase();
    return is429(err) || /service unavailable|internal error|temporarily/i.test(msg);
}

// ═══════════════════════════════════════════════════════════════════════════
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

// Parse overlay options from a request body. Honors brand default when the
// client didn't explicitly send `overlayEnabled`. Returns null if the brand
// doesn't support overlays (so generateShot short-circuits cheaply).
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

// Brand-aware shot catalog. Merges the shared catalog with any extraShots the
// active brand declares (e.g. Taheri's taheri_signature walnut-on-emerald shot).
function buildShotCatalog(brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    const merged = { ...SHOT_CATALOG };
    for (const extra of brand.extraShots || []) {
        // The scenePrompt lives in buildShotPrompt's scenes lookup; only meta here.
        const { scenePrompt, ...meta } = extra;
        merged[extra.id] = meta;
    }
    return merged;
}

// ── Prompt builders per shot ───────────────────────────────────────────────
function buildShotPrompt(shotId, customInstruction, hasAnchor = false, autoMatchRing = false, multiPiece = false, brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    const base = brand.baseIntro;

    // Only when the user has flagged that the folder contains multiple distinct pieces.
    // Without this flag we assume one piece per folder (the default, and how most users organise).
    const primaryBlock = multiPiece
        ? `\nMULTI-PIECE FOLDER — USER-DECLARED: The user has indicated this folder contains reference photos for MORE THAN ONE distinct jewelry piece (e.g., a bangle AND a matching ring, or a necklace AND earrings as a set). Pick the PRIMARY piece using these rules in order:
  1. The piece that appears in the MAJORITY of the reference images is the primary subject.
  2. If two pieces appear in equal numbers, the LARGER / MORE STRUCTURAL piece is primary (a bangle outranks a ring; a necklace outranks earrings; a maang tikka outranks small studs).
  3. Product shots (hero, angle, detail, flat, stand) must show ONLY the primary piece — do NOT substitute a companion piece as the subject.
  4. Model shots show only the primary piece worn on the appropriate body part.
  5. Group / flat-lay shots MAY include companion pieces as secondary decorative context, but the primary piece must remain the clear hero at larger size and more prominent positioning.

CRITICAL: do NOT let a smaller companion piece become the subject. If a bangle and a ring both appear in the references, the bangle is the subject — the ring is context at most. If a necklace and earrings both appear, the necklace is the subject — earrings are context at most.

If the user's ADDITIONAL DIRECTION (below) names a specific piece to focus on (e.g., "focus on the ring"), that override wins — use the named piece as the primary subject.
`
        : '';

    // When an anchor reference is present, add IP-Adapter-style consistency conditioning
    const anchorBlock = hasAnchor
        ? `\nCONSISTENCY ANCHOR: The LAST reference image is a clean studio product shot I already generated of this exact jewelry piece. Treat it as your visual ground truth. Every stone count, every prong, every metal tone, every proportion in your output MUST match this anchor image exactly. If there is any ambiguity between the raw reference photos and the anchor, defer to the anchor — it is the canonical representation of this piece.\n`
        : '';

    const scenes = {
        ecom_hero: `SCENE: Professional ecommerce hero shot. Pure white (#FFFFFF) seamless background. The jewelry is centered, occupying approximately 65% of the frame. Camera is at a slight elevation (15–20°) to show the decorative face. Even, diffused studio lighting from two softboxes at 45° angles, creating clean specular highlights on metal surfaces and brilliant stone reflections. Subtle drop shadow beneath the piece for grounding. No props, no distractions — the piece is the entire composition.
CAMERA: 100mm macro lens, f/8, focus-stacked for edge-to-edge sharpness. Color-accurate white balance (5500K). Shot on medium format digital for maximum detail.`,

        ecom_angle: `SCENE: Three-quarter angle product shot. Pure white (#FFFFFF) seamless background. Camera positioned at 45° to the front face, slightly elevated (20–25°), revealing the depth, profile, and side construction of the piece. This angle shows how the jewelry looks in three dimensions — the curve of a bangle, the height of a setting, the thickness of metalwork. Same even studio lighting with clean highlights.
CAMERA: 100mm macro, f/8, focus-stacked. The viewer should feel they can reach in and pick up the piece.`,

        ecom_detail: `SCENE: Extreme macro close-up. Camera is 1–3 cm from the most intricate area of the jewelry — the center stone and its setting, the finest filigree, or the most detailed metalwork. Fill the entire frame with this detail. Pure white (#FFFFFF) background and surface beneath. Razor-sharp focus on the subject with natural bokeh softening the edges. This shot reveals craftsmanship — individual prongs, stone facets, metal grain, pavé precision.
CAMERA: Dedicated macro lens at 1:1 magnification, f/5.6 for shallow depth, ring light for even illumination without harsh shadows. Shot so close the viewer can count individual stones.`,

        ecom_flat: `SCENE: Overhead flat lay on pure white (#FFFFFF) surface and background. Camera directly above (90° bird's eye). The jewelry is laid flat, centered, with its decorative face pointing up. For bangles/bracelets: circular shape fully visible. For necklaces: arranged in an elegant drape or gentle curve. For rings: face up, slightly angled. Even, shadowless lighting from a large overhead softbox. Clean, minimal, editorial.
CAMERA: 85mm, f/8, tripod-mounted directly overhead. Perfect symmetry in composition.`,

        ecom_stand: `SCENE: ${brand.ecomStandBrandRef}. Look at the reference photo(s) to determine the jewelry type, then choose the CORRECT display:
- Bangles / cuffs / bracelets: upright on a velvet cushion roll or half-cylinder stand, resting naturally with the decorative face toward camera. NEVER use a T-bar or hanging stand for bangles.
- Rings: on a slim velvet cone or small cushion, tilted slightly toward camera.
- Necklaces / chokers: draped over a fabric neck bust or laid on a velvet tray in an elegant curve.
- Earrings: on a small padded earring card or low T-bar stand.
- Maang tikka / headpieces: laid flat on a velvet tray or silk fabric.

The stand/display is elegant and minimal, in matte cream, soft gold, or deep velvet. Background is pure white (#FFFFFF). Soft, warm window-style light from the upper left creates gentle shadows and a luxurious mood. The decorative face of the jewelry faces the camera. The display should look natural — the jewelry should sit the way it would in a real boutique.
CAMERA: 85mm f/2.8, slightly shallow depth of field to separate the piece from the background. Warm color temperature (5800K).`,

        ecom_group: `SCENE: Scale and context shot. Pure white (#FFFFFF) background and surface. The jewelry is placed alongside a subtle, universally understood size reference — a single fresh rose petal, a small velvet pouch, or an elegant hand mirror. The reference object is secondary and slightly out of focus. The jewelry remains the hero. This shot communicates real-world scale and presence.
CAMERA: 85mm, f/4, with the jewelry in sharp focus and the reference object in soft focus behind or beside it.`,

        model_wrist: `SCENE: The jewelry is worn on the wrist/hand of a model. For bangles and bracelets: worn snugly on the wrist, sitting flush against skin with the outer decorative face toward the camera. For rings: worn on the ring finger, hand relaxed.

PROPORTION & SCALE — CRITICAL, DO NOT GET THIS WRONG:
The reference photo shows the TRUE physical size of this piece. You MUST preserve that scale against the model's anatomy. Anchor measurements for an adult woman's hand (use these as ground truth):
- Wrist circumference: ~15–17 cm (inner diameter of a bangle that fits = roughly 58–65 mm across).
- Back-of-hand width (knuckles): ~8 cm.
- Ring finger width at base: ~15–17 mm.
- Palm length (wrist crease to middle-finger base): ~10 cm.

Rules:
1. A standard bangle fits ONLY over the knuckles — its inner diameter is barely larger than the widest part of the hand. It should NOT look like a giant hoop dwarfing the wrist, and it should NOT look like a tiny ring strangling the wrist. Inner diameter must read as ~60 mm against a ~8 cm knuckle width — i.e. the bangle opening is about 3/4 the width of the knuckles when seen straight-on.
2. A cuff bracelet's width (top-to-bottom face height) is typically 10–30 mm. It covers a fraction of the wrist, not the entire forearm. Do NOT stretch it into a gauntlet.
3. A tennis/chain bracelet sits loose on the wrist with a small gap — do not inflate its links.
4. A ring covers ~15–20 mm of finger length at most. It should not cover two finger segments unless the reference shows that.
5. Metal thickness, stone size, and engraving depth must match the reference at 1:1 — do NOT thicken the profile "to make it pop."
6. If the reference photo includes a hand for scale, measure the piece against the fingers/wrist in the reference and reproduce the exact ratio.

FITTING: Study the reference photo(s) carefully. If the jewelry is shown being worn, replicate the EXACT fit — how loose or tight it sits on the body, how much it slides or grips, the gap between jewelry and skin, the way it drapes or hangs under gravity. A loose bangle must look loose; a snug cuff must look snug. Match the reference fit precisely.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, natural skin texture, clean manicure with nude or soft pink nails
- Adult proportions — normal-sized hand, not an oversized or undersized hand
- Arm extended forward, elbow slightly bent
- Wrist level or slightly lowered, fingers pointing DOWNWARD and loosely relaxed
- Back of hand faces the camera, palm faces away
- Do NOT raise the hand with fingers pointing up, do NOT show the palm

FRAMING: Tight crop showing only the hand, wrist, and a few inches of forearm. No face, no shoulder, no torso. Do NOT zoom in so tight that the piece fills the frame and loses anatomical context — keep the whole hand visible so scale is readable.
LIGHTING: Single soft key light from above-left, warm neutral blurred backdrop (creamy beige or soft gold), 85mm f/1.4 equivalent depth of field. Subtle film grain. The skin should glow warmly.`,

        model_neck: `SCENE: The jewelry is worn around the neck of a model. The necklace or choker sits naturally on the collarbone/décolletage area.

PROPORTION & SCALE — CRITICAL, DO NOT GET THIS WRONG:
Anchor measurements for an adult woman's neck/décolletage (ground truth):
- Neck circumference: ~32–36 cm.
- Collarbone-to-collarbone span (suprasternal notch width): ~14–17 cm.
- Vertical distance from jaw to collarbone: ~10–12 cm.
- Chin-to-sternum: ~18–22 cm.

Rules:
1. A CHOKER (38–42 cm) sits on the mid-neck with no gap.
2. A PRINCESS necklace (44–48 cm) rests on the collarbones.
3. A MATINEE (55–60 cm) hangs to mid-chest.
4. A pendant diameter of 2 cm should read roughly 1/8 of the collarbone-to-collarbone span — do NOT inflate pendants to fill the chest.
5. Chain link gauge and pendant stone sizes must match the reference 1:1. Do NOT thicken chains or enlarge stones.
6. If the reference shows the piece worn, match that drape length and fit exactly.

FITTING: Study the reference photo(s) carefully. If the jewelry is shown being worn, replicate the EXACT fit — how loose or tight it sits on the neck, the drape length, the gap between the piece and skin, whether it sits high on the throat or low on the collarbone. Match the reference fit precisely.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, natural skin, elegant bone structure
- Adult proportions — normal-sized neck and shoulders, not elongated or miniaturized
- Head tilted very slightly to one side, chin slightly lifted
- Wearing a simple, solid-color top or bare shoulders (nothing competing with the jewelry)
- Hair pulled back or swept to one side to fully reveal the necklace

FRAMING: From mid-chest to just below the chin. The necklace is the clear focal point. Jawline and neck visible for context but the jewelry dominates. Keep enough anatomy visible for scale to read correctly.
LIGHTING: Soft, warm key light from above-right, gentle fill from the left. Warm neutral backdrop. 85mm f/1.8, shallow depth. The skin glows, the metal catches light beautifully.`,

        model_ear: `SCENE: The earring is worn on the ear of a model. Close-up of the ear, jawline, and a hint of neck.

PROPORTION & SCALE — CRITICAL, DO NOT GET THIS WRONG:
Anchor measurements for an adult woman's ear (ground truth):
- Ear total height (top of helix to bottom of lobe): ~6–6.5 cm.
- Earlobe height: ~1.5–2 cm.
- Earlobe width: ~1–1.5 cm.
- Ear-to-jawline vertical distance: ~5–7 cm.

Rules:
1. A small STUD is 5–8 mm — it occupies only the centre of the earlobe, not the whole lobe.
2. A MID-DROP earring (hoops, small jhumkas) is 20–30 mm total drop — bottom sits roughly level with the earlobe or just below.
3. A LONG CHANDELIER / jhumka is 40–70 mm total drop — swings to the jawline, not past it unless the reference shows that.
4. Stone diameters, pearl sizes and filigree thickness must match the reference 1:1. Do NOT scale the piece up to "fill" the ear.
5. If the reference shows the earring worn, reproduce the exact drop length relative to the earlobe/jawline.

FITTING: Study the reference photo(s) carefully. If the earring is shown being worn, replicate the EXACT fit — how it hangs, the drop length, whether it sits close to the lobe or dangles freely, and how it moves with gravity. Match the reference fit precisely.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, clean skin, elegant jawline
- Adult proportions — normal-sized ear and jaw, not oversized or undersized
- Head turned slightly (three-quarter profile) to present the ear naturally
- Hair tucked behind the ear or swept up to fully reveal the earring
- Expression serene, mouth relaxed (if lips are visible at edge of frame)

FRAMING: Tight crop on the ear and surrounding area. The earring is the clear hero. Show enough of the jaw and neck for anatomical context so the viewer can read the earring's real-world size.
LIGHTING: Soft key light from the front-left, gentle rim light to separate from background. Warm blurred backdrop. 100mm f/2, very shallow depth — the earring is razor-sharp, everything else falls off softly.`,

        model_lifestyle: `SCENE: Lifestyle editorial shot. The model is wearing the jewelry in a warm, luxurious setting — think golden hour light, soft furnishings, or a beautiful window. The mood is aspirational, elegant, and distinctly South Asian-luxe.

PROPORTION & SCALE — CRITICAL, DO NOT GET THIS WRONG:
At a medium framing distance the jewelry should read as a natural-sized piece against adult anatomy, NOT as an enlarged hero element. Use the same anchor measurements as for model_wrist / model_neck / model_ear (bangle ~60 mm inner diameter; princess necklace 44–48 cm; long earring drop 40–70 mm). The piece appears smaller in this wider frame than in close-ups — do NOT zoom or inflate it to compensate. Stone sizes, chain gauges, and metal thickness must match the reference 1:1.

FITTING: Study the reference photo(s) carefully. If the jewelry is shown being worn, replicate the EXACT fit — how loose or tight it sits on the body, the drape, the gap between jewelry and skin, the way it hangs under gravity. Match the reference fit precisely.

MODEL & POSE:
- Woman in her early 20s, warm South Asian skin tone, styled beautifully but not overly made up
- Adult proportions throughout — no elongated neck, oversized hands, or miniaturised torso
- Natural, candid-feeling pose — adjusting the jewelry, looking away from camera, or mid-movement
- Wearing complementary but simple clothing that doesn't compete (solid colors, elegant draping)

FRAMING: Medium shot (waist up or three-quarter). The jewelry should be clearly visible and prominent despite the wider framing, but sized correctly for the distance — NOT enlarged to fake prominence.
LIGHTING: Warm, natural-feeling light (golden hour or large window). Slight haze or warmth in the atmosphere. 50mm f/1.8, cinematic depth of field. Slight film grain for editorial feel.`,

        marble: `SCENE: Luxury surface shot. The jewelry rests on a white or cream Carrara marble surface with soft, natural grey veining. Beside the jewelry (not touching): one or two minimal props — a small sprig of dried flowers, a fragment of silk ribbon, or a tiny gold-rimmed dish. Props are muted and secondary. The composition is editorial, airy, and luxurious.
LIGHTING: Soft, warm natural light from a large window to the left. Gentle shadows. The marble surface has a slight sheen. Warm color palette overall.
CAMERA: 45° angle, 85mm f/2.8, the jewelry is in perfect focus, props and marble veining fall off softly. The feeling is a luxury magazine editorial spread.`,

        marble_dark: `SCENE: Dramatic dark surface shot. The jewelry rests on dark emperador or nero marquina marble — deep brown-black with gold or white veining. The mood is dramatic, moody, and high-end. Minimal props if any — perhaps a single dark velvet fold or a matte black box edge barely visible. The jewelry catches all the light and pops against the dark surface.
LIGHTING: Single focused light source from above-right, creating dramatic highlights on the metal and stones while the marble stays dark and moody. Deep shadows, high contrast. The gold of the jewelry glows against the darkness.
CAMERA: Low angle (15–20°), 100mm f/2.8, shallow depth. Cinematic, editorial, powerful. Think luxury brand campaign.`,
    };

    // Merge in brand-specific extra shots (e.g. Taheri's taheri_signature).
    for (const extra of brand.extraShots || []) {
        if (extra.scenePrompt) scenes[extra.id] = extra.scenePrompt;
    }

    const scene = scenes[shotId] || scenes.ecom_hero;

    const ringMatchBlock = autoMatchRing
        ? `\nCOMPLEMENTARY RING PAIRING — CONDITIONAL:
First, audit ALL of the reference photos provided (every raw reference AND every inspo/inspiration image — scan every single one before deciding). Determine two things:
  1. Is the primary piece a bracelet, bangle, or cuff?
  2. Does a ring appear ANYWHERE across the full set of reference images (worn on a finger, laid beside the bracelet, on a stand, in a flat-lay, even partially visible in the corner of an inspo shot)?

Decision rules — apply in order, stop at the first match:
  • If the primary piece is NOT a bracelet/bangle/cuff → IGNORE this entire instruction. Do nothing. Do not add any ring.
  • If a ring IS visible in ANY reference image (raw OR inspo, any angle, any prominence) → IGNORE this entire instruction. The user has already specified the ring; do not invent a different one, do not add an additional ring. Reproduce that existing ring faithfully if it would naturally appear in the scene.
  • ONLY if the primary piece IS a bracelet/bangle/cuff AND zero rings appear in any of the reference images → design a coordinated matching ring and include it in this output.

WHEN YOU DESIGN THE RING — HARD CONSTRAINTS:
The ring MUST be a faithful MINIATURIZATION of the bracelet's own design, not merely "coordinated" or "from the same collection". Treat it as if a jeweler literally shrank the bracelet's decorative face down to fit on a finger. Specifically:
  • Transplant the bracelet's central motif / focal decorative element (the largest stone cluster, the main floral/geometric/paisley motif, the dominant filigree shape) onto the ring face. The motif becomes the ring's face, reduced in scale to fit a finger (~14–20 mm wide).
  • Exact same metal tone, finish, and surface treatment (matte vs polished, yellow vs rose vs white gold, antiqued vs bright).
  • Exact same stone material, color, cut, and setting technique (pavé, bezel, prong, kundan, polki, meenakari enamel — whatever the bracelet uses).
  • Exact same ornamental language (beading, milgrain, granulation, engraving patterns) — if the bracelet has filigree curls, the ring has filigree curls at a smaller scale.
  • The ring's shank/band is a simple slim version of the bracelet's band — same metal, same width proportion relative to the motif, no foreign design elements.
  • Omit bracelet-specific structural parts (hinges, clasps, side wings, the full bangle circumference) — only the decorative face survives the shrink.
  • Result test: a viewer should immediately say "that ring is clearly the matching miniature of this bracelet" at a single glance, not "that ring pairs nicely with this bracelet".

Place it appropriately for the scene: for ecommerce/product shots, position the ring elegantly next to the bracelet on the same surface; for model shots of the wrist/hand, place the ring on the ring finger of the same hand wearing the bracelet.

When in doubt, DO NOT add a ring. Adding an unwanted ring is a worse failure than omitting one.
`
        : '';

    const parts = [
        base,
        primaryBlock,
        anchorBlock,
        ringMatchBlock,
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
app.get('/shots', (req, res) => {
    const brandId = BRANDS[req.query.brand] ? req.query.brand : DEFAULT_BRAND;
    res.json(buildShotCatalog(brandId));
});
app.get('/brands', (_req, res) => res.json({ brands: listBrands(), defaultBrand: DEFAULT_BRAND }));
app.get('/usage', (_req, res) => res.json(usageStats));
app.post('/usage/reset', (_req, res) => {
    for (const key of Object.keys(usageStats.session)) {
        usageStats.session[key] = { images: 0, cost: 0 };
    }
    usageStats.history = [];
    try { fs.writeFileSync(USAGE_STATS_PATH, JSON.stringify(usageStats, null, 2)); } catch (e) {}
    res.json({ reset: true });
});
app.get('/cost-rates', (_req, res) => res.json(COST_PER_IMAGE));

// ── Concurrency control (#18) ─────────────────────────────────────────────
app.get('/concurrency', (_req, res) => {
    res.json({ maxConcurrent: MAX_CONCURRENT, productConcurrency: PRODUCT_CONCURRENCY });
});
app.post('/concurrency', (req, res) => {
    if (req.body.maxConcurrent !== undefined) {
        MAX_CONCURRENT = Math.max(1, Math.min(6, parseInt(req.body.maxConcurrent, 10) || MAX_CONCURRENT));
    }
    if (req.body.productConcurrency !== undefined) {
        PRODUCT_CONCURRENCY = Math.max(1, Math.min(6, parseInt(req.body.productConcurrency, 10) || PRODUCT_CONCURRENCY));
    }
    res.json({ maxConcurrent: MAX_CONCURRENT, productConcurrency: PRODUCT_CONCURRENCY });
});

// ── Scrape product images from URL ─────────────────────────────────────────
app.post('/scrape-url', async (req, res) => {
    const { url } = req.body;
    if (!url) return res.status(400).json({ error: 'No URL provided.' });

    try {
        // Extract product handle from Shopify URL: /products/{handle}
        const parsed = new URL(url);
        const match = parsed.pathname.match(/\/products\/([^/?#]+)/);
        if (!match) return res.status(400).json({ error: 'Could not find product handle in URL. Expected format: /products/{product-name}' });

        const handle = match[1];
        const jsonUrl = `${parsed.origin}/products/${handle}.json`;
        console.log(`[Scrape] Fetching ${jsonUrl}`);

        const response = await fetch(jsonUrl);
        if (!response.ok) return res.status(400).json({ error: `Failed to fetch product JSON (HTTP ${response.status})` });

        const data = await response.json();
        const product = data.product;
        if (!product) return res.status(400).json({ error: 'No product data found.' });

        const images = product.images || [];
        if (images.length === 0) return res.status(400).json({ error: 'Product has no images.' });

        console.log(`[Scrape] "${product.title}" — ${images.length} image(s)`);

        // Download each image and convert to base64
        const results = await Promise.all(images.map(async (img, i) => {
            const imgRes = await fetch(img.src);
            if (!imgRes.ok) return null;
            const buf = Buffer.from(await imgRes.arrayBuffer());
            const ext = path.extname(new URL(img.src).pathname).toLowerCase() || '.jpg';
            const mimeMap = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif' };
            const mime = mimeMap[ext] || 'image/jpeg';
            return { base64: buf.toString('base64'), mimeType: mime, name: `${handle}_${i + 1}${ext}`, width: img.width, height: img.height };
        }));

        const valid = results.filter(Boolean);
        console.log(`[Scrape] Downloaded ${valid.length} image(s)`);
        res.json({ success: true, title: product.title, images: valid });
    } catch (err) {
        console.error('[Scrape Error]', err.message);
        res.status(500).json({ error: err.message });
    }
});

// ── Serve generated files from disk ─────────────────────────────────────────
app.get('/file', (req, res) => {
    const filePath = req.query.path;
    if (!filePath || !fs.existsSync(filePath)) return res.status(404).send('Not found');
    res.sendFile(path.resolve(filePath));
});

// Return the first reference image from a product folder (largest = best ref)
app.get('/product/reference', (req, res) => {
    const folder = req.query.folder;
    if (!folder || !fs.existsSync(folder)) return res.status(404).json({ error: 'Folder not found' });
    try {
        const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
        const files = fs.readdirSync(folder)
            .filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.'))
            .map(f => {
                const full = path.join(folder, f);
                let size = 0;
                try { size = fs.statSync(full).size; } catch (e) {}
                return { full, size };
            })
            .sort((a, b) => b.size - a.size);
        if (files.length === 0) return res.status(404).json({ error: 'No images' });
        res.json({ path: files[0].full });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ── Batch cancellation state ────────────────────────────────────────────────
let activeBatchId  = null;
let batchCancelled = false;
let skipProducts   = new Set();   // product names to skip mid-run
let batchAbortController = null;  // aborts in-flight API calls & retry sleeps

class CancelledError extends Error {
    constructor() { super('CANCELLED'); this.name = 'CancelledError'; this.cancelled = true; }
}
function isCancelled(err) {
    return err && (err.cancelled === true || err.name === 'CancelledError' || err.message === 'CANCELLED');
}
function throwIfCancelled() {
    if (batchCancelled) throw new CancelledError();
}
// Race any promise against the current batch abort signal so cancel returns control instantly.
function withCancel(promise) {
    if (!batchAbortController) return promise;
    const signal = batchAbortController.signal;
    if (signal.aborted) return Promise.reject(new CancelledError());
    return new Promise((resolve, reject) => {
        const onAbort = () => reject(new CancelledError());
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(
            v => { signal.removeEventListener('abort', onAbort); resolve(v); },
            e => { signal.removeEventListener('abort', onAbort); reject(e); }
        );
    });
}
function abortableSleep(ms) {
    return new Promise((resolve, reject) => {
        const t = setTimeout(resolve, ms);
        if (batchAbortController) {
            const signal = batchAbortController.signal;
            const onAbort = () => { clearTimeout(t); reject(new CancelledError()); };
            if (signal.aborted) return onAbort();
            signal.addEventListener('abort', onAbort, { once: true });
        }
    });
}

const cancelHandler = (req, res) => {
    if (activeBatchId) {
        batchCancelled = true;
        if (batchAbortController) batchAbortController.abort();
        // Flush the concurrency queue so pending acquires reject immediately
        while (geminiQueue.length) {
            const entry = geminiQueue.shift();
            const rej = typeof entry === 'function' ? null : entry?.reject;
            if (rej) rej(new CancelledError());
        }
        console.log('[Batch] ⚠ Cancel requested — aborting in-flight calls.');
        res.json({ cancelled: true });
    } else {
        res.json({ cancelled: false, message: 'No active batch.' });
    }
};
app.post('/cancel-batch', cancelHandler);
app.post('/batch/cancel', cancelHandler);

// Mid-run: skip a single in-flight/queued product
app.post('/batch/skip-product', express.json(), (req, res) => {
    const name = (req.body && req.body.product) || '';
    if (!name) return res.status(400).json({ error: 'Missing product name.' });
    if (!activeBatchId) return res.json({ skipped: false, message: 'No active batch.' });
    skipProducts.add(name);
    res.json({ skipped: true, product: name });
});

// Pre-run: list subfolders with image counts so the user can choose which to process
app.get('/scan-folder', (req, res) => {
    const folderPath = (req.query.folderPath || '').trim().replace(/^['"]|['"]$/g, '');
    if (!folderPath) return res.status(400).json({ error: 'No folder path provided.' });
    if (!fs.existsSync(folderPath)) return res.status(404).json({ error: `Folder not found: ${folderPath}` });
    if (!fs.statSync(folderPath).isDirectory()) return res.status(400).json({ error: 'That path is a file, not a folder.' });

    const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
    try {
        const dirs = fs.readdirSync(folderPath, { withFileTypes: true })
            .filter(d => d.isDirectory() && !d.name.startsWith('.') && d.name !== 'ecommerce' && d.name !== 'output')
            .map(d => {
                const full = path.join(folderPath, d.name);
                let imageCount = 0;
                try {
                    imageCount = fs.readdirSync(full).filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.')).length;
                } catch (e) {}
                return { name: d.name, imageCount };
            });
        res.json({ folderPath, subfolders: dirs });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ── Single product generation ───────────────────────────────────────────────
app.post('/generate', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const shotIds           = JSON.parse(req.body.shots || '[]');
    const customInstruction = (req.body.customInstruction || '').trim() || null;
    const provider          = (req.body.provider || 'gemini').trim();
    const resolution        = ['draft', 'standard', 'high'].includes(req.body.resolution) ? req.body.resolution : 'standard';
    const autoMatchRing     = req.body.autoMatchRing === '1' || req.body.autoMatchRing === 1 || req.body.autoMatchRing === true;
    const multiPiece        = req.body.multiPiece === '1' || req.body.multiPiece === 1 || req.body.multiPiece === true;
    const brandId           = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
    const overlayOpts       = parseOverlayOpts(req.body, brandId);
    const shotCatalog       = buildShotCatalog(brandId);

    if (shotIds.length === 0) return res.status(400).json({ error: 'No shots selected.' });

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        console.log(`[Generate] ${shotIds.length} shot(s) via ${provider} @${resolution}: ${shotIds.join(', ')}`);

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
        const anchorShot = shotCatalog[anchorId];
        const anchorResult = await generateShot(anchorId, imageInputs, customInstruction, false, provider, autoMatchRing, multiPiece, resolution, false, brandId, overlayOpts);
        anchorRef = { base64: anchorResult.base64, mimeType: 'image/png' };
        results.push({ id: anchorId, label: anchorShot.label, category: anchorShot.category, data: anchorResult.base64, provider: anchorResult.provider });

        // Generate all remaining shots in parallel, ALL receiving the anchor
        const remaining = shotIds.filter(id => id !== anchorId);

        if (remaining.length > 0) {
            const refsWithAnchor = [...imageInputs, anchorRef];
            const parallel = await Promise.all(remaining.map(async (shotId) => {
                const shot = shotCatalog[shotId];
                if (!shot) return null;
                const result = await generateShot(shotId, refsWithAnchor, customInstruction, true, provider, autoMatchRing, multiPiece, resolution, false, brandId, overlayOpts);
                return { id: shotId, label: shot.label, category: shot.category, data: result.base64, provider: result.provider };
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
    const resolution        = ['draft', 'standard', 'high'].includes(req.body.resolution) ? req.body.resolution : 'standard';
    const autoMatchRing     = req.body.autoMatchRing === '1' || req.body.autoMatchRing === 1 || req.body.autoMatchRing === true;
    const multiPiece        = req.body.multiPiece === '1' || req.body.multiPiece === 1 || req.body.multiPiece === true;
    const brandId           = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
    const overlayOpts       = parseOverlayOpts(req.body, brandId);

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        const shotCatalog = buildShotCatalog(brandId);
        const shot = shotCatalog[shotId];
        if (!shot) return res.status(400).json({ error: 'Unknown shot type.' });

        const result = await generateShot(shotId, imageInputs, customInstruction, false, provider, autoMatchRing, multiPiece, resolution, false, brandId, overlayOpts);
        res.json({ success: true, imageData: result.base64, provider: result.provider, usage: usageStats });
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
    res.setHeader('X-Accel-Buffering', 'no');   // disable proxy buffering (nginx etc.)
    res.flushHeaders();

    // Disable socket timeout for long-running SSE streams
    req.socket.setTimeout(0);
    req.socket.setNoDelay(true);
    req.socket.setKeepAlive(true, 30000);

    // Track client disconnection
    let clientDisconnected = false;
    req.on('close', () => { clientDisconnected = true; });

    const send = (payload) => {
        if (clientDisconnected) return;
        try {
            res.write(`data: ${JSON.stringify(payload)}\n\n`);
        } catch (e) {
            clientDisconnected = true;
            console.error('[SSE] Write failed, client likely disconnected');
        }
    };

    // SSE heartbeat — send a comment every 15 seconds to keep connection alive
    const heartbeat = setInterval(() => {
        if (clientDisconnected) { clearInterval(heartbeat); return; }
        try {
            res.write(': heartbeat\n\n');
        } catch (e) {
            clientDisconnected = true;
            clearInterval(heartbeat);
        }
    }, 15000);

    const folderPath        = (req.query.folderPath || '').trim().replace(/^['"]|['"]$/g, '');
    const customInstruction = (req.query.customInstruction || '').trim() || null;
    const shotIds           = JSON.parse(req.query.shots || '[]');
    const provider          = (req.query.provider || 'gemini').trim();
    const resolution        = ['draft', 'standard', 'high'].includes(req.query.resolution) ? req.query.resolution : 'standard';
    const resume            = req.query.resume === '1';
    const autoMatchRing     = req.query.autoMatchRing === '1';
    const multiPiece        = req.query.multiPiece === '1';
    const brandId           = BRANDS[req.query.brand] ? req.query.brand : DEFAULT_BRAND;
    const overlayOpts       = parseOverlayOpts(req.query, brandId);
    const shotCatalog       = buildShotCatalog(brandId);
    // Pre-run skip list: JSON array of subfolder names the user unchecked
    let skipFolders = [];
    try { skipFolders = JSON.parse(req.query.skipFolders || '[]'); } catch (e) { skipFolders = []; }
    const skipFoldersSet = new Set(skipFolders);

    if (!folderPath) { send({ type: 'error', message: 'No folder path provided.' }); return res.end(); }
    if (!fs.existsSync(folderPath)) { send({ type: 'error', message: `Folder not found: ${folderPath}` }); return res.end(); }
    if (!fs.statSync(folderPath).isDirectory()) { send({ type: 'error', message: 'That path is a file, not a folder.' }); return res.end(); }
    if (shotIds.length === 0) { send({ type: 'error', message: 'No shots selected.' }); return res.end(); }

    const allDirs = fs.readdirSync(folderPath, { withFileTypes: true })
        .filter(d => d.isDirectory() && !d.name.startsWith('.') && d.name !== 'ecommerce' && d.name !== 'output')
        .map(d => ({ name: d.name, fullPath: path.join(folderPath, d.name) }));

    const productDirs = allDirs.filter(d => !skipFoldersSet.has(d.name));
    const excludedCount = allDirs.length - productDirs.length;

    if (allDirs.length === 0) {
        send({ type: 'error', message: 'No product subfolders found.' });
        return res.end();
    }
    if (productDirs.length === 0) {
        send({ type: 'error', message: 'All subfolders were excluded — nothing to process.' });
        return res.end();
    }

    activeBatchId = Date.now().toString();
    batchCancelled = false;
    batchAbortController = new AbortController();
    skipProducts = new Set();
    if (excludedCount > 0) {
        console.log(`[Batch] Excluding ${excludedCount} folder(s): ${skipFolders.join(', ')}`);
    }

    send({ type: 'start', total: productDirs.length, batchId: activeBatchId, shots: shotIds, resume, productConcurrency: PRODUCT_CONCURRENCY });
    console.log(`[Batch] Starting — ${productDirs.length} product(s), ${PRODUCT_CONCURRENCY} in parallel, shots: ${shotIds.join(', ')}`);

    // Process a single product (extracted so we can run N in parallel)
    async function processProduct(productName, productFolder) {
        if (clientDisconnected || batchCancelled) return;
        if (skipProducts.has(productName)) {
            send({ type: 'product_skipped', product: productName, reason: 'User skipped before start' });
            console.log(`[Product] ⊘ Skipped (pre-start): ${productName}`);
            return;
        }

        send({ type: 'product_start', product: productName, productFolder });
        console.log(`\n[Product] ▶ ${productName}`);

        try {
            const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
            // Sort by file size desc — largest files (usually highest resolution) come first,
            // so the anchor-first pipeline picks the best reference.
            const imageFiles = fs.readdirSync(productFolder)
                .filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.'))
                .map(f => {
                    const full = path.join(productFolder, f);
                    let size = 0;
                    try { size = fs.statSync(full).size; } catch (e) {}
                    return { full, size };
                })
                .sort((a, b) => b.size - a.size)
                .map(x => x.full);

            if (imageFiles.length === 0) {
                send({ type: 'product_error', product: productName, message: 'No images found in folder.' });
                send({ type: 'product_done', product: productName });
                return;
            }

            const imageInputs = await Promise.all(imageFiles.map(async (fp) => {
                const raw = await fs.promises.readFile(fp);
                const buf = await toJpeg(fp, raw);
                return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
            }));

            const outDir = path.join(folderPath, 'output', productName);
            fs.mkdirSync(outDir, { recursive: true });

            // ── Resume: check if ALL shots for this product already exist ──
            if (resume) {
                const allDone = shotIds.every(id => fs.existsSync(path.join(outDir, `${id}.png`)));
                if (allDone) {
                    for (const shotId of shotIds) {
                        const shot = shotCatalog[shotId];
                        if (!shot) continue;
                        const outPath = path.join(outDir, `${shotId}.png`);
                        send({ type: 'angle_skipped', product: productName, angle: shotId, label: shot.label, savedTo: outPath });
                    }
                    send({ type: 'product_done', product: productName, skipped: true });
                    return;
                }
            }

            // ── Anchor-first consistency pipeline ──
            const anchorId = shotIds.includes('ecom_hero') ? 'ecom_hero'
                : shotIds.find(id => id.startsWith('ecom_'))
                || shotIds[0];

            let anchorRef = null;
            let anchorGenerated = false;
            const anchorShot = shotCatalog[anchorId];
            const anchorOutPath = path.join(outDir, `${anchorId}.png`);

            // Resume: try to reuse existing anchor
            if (resume && fs.existsSync(anchorOutPath)) {
                console.log(`[Resume] Reusing existing anchor for ${productName}/${anchorId}`);
                const existingBuf = await fs.promises.readFile(anchorOutPath);
                anchorRef = { base64: existingBuf.toString('base64'), mimeType: 'image/png' };
                send({ type: 'angle_skipped', product: productName, angle: anchorId, label: `${anchorShot.label} (anchor)`, savedTo: anchorOutPath });
            } else {
                send({ type: 'angle_start', product: productName, angle: anchorId, label: `${anchorShot.label} (anchor)` });
                console.log(`  [Shot] Generating anchor: ${anchorShot.label} (${anchorId}) via ${provider}...`);
                try {
                    const result = await generateShot(anchorId, imageInputs, customInstruction, false, provider, autoMatchRing, multiPiece, resolution, !resume, brandId, overlayOpts);
                    anchorRef = { base64: result.base64, mimeType: 'image/png' };
                    fs.writeFileSync(anchorOutPath, Buffer.from(result.base64, 'base64'));
                    anchorGenerated = true;
                    send({ type: 'angle_done', product: productName, angle: anchorId, label: anchorShot.label, savedTo: anchorOutPath, provider: result.provider });
                    send({ type: 'usage', usage: usageStats });
                    console.log(`  [Shot] ✓ ${anchorShot.label} → ${anchorOutPath}`);
                } catch (err) {
                    if (isCancelled(err) || batchCancelled) {
                        console.log(`  [Shot] ⊘ Cancelled: ${anchorShot.label}`);
                    } else {
                        send({ type: 'angle_error', product: productName, angle: anchorId, message: err.message });
                        console.error(`  [Shot] ✗ ${anchorShot.label}: ${err.message}`);
                    }
                }
            }

            if (batchCancelled) {
                send({ type: 'product_done', product: productName });
                return;
            }
            if (skipProducts.has(productName)) {
                send({ type: 'product_skipped', product: productName, reason: 'User skipped after anchor' });
                console.log(`[Product] ⊘ Skipped (after anchor): ${productName}`);
                return;
            }

            // All remaining shots get the anchor reference
            const remaining = shotIds.filter(id => id !== anchorId);
            const refsWithAnchor = anchorRef ? [...imageInputs, anchorRef] : imageInputs;
            const hasAnchor = !!anchorRef;

            // Figure out which remaining shots need generating vs skipping
            const toGenerate = [];
            for (const shotId of remaining) {
                const shot = shotCatalog[shotId];
                if (!shot) continue;
                const shotOutPath = path.join(outDir, `${shotId}.png`);
                if (resume && fs.existsSync(shotOutPath)) {
                    send({ type: 'angle_skipped', product: productName, angle: shotId, label: shot.label, savedTo: shotOutPath });
                } else {
                    send({ type: 'angle_start', product: productName, angle: shotId, label: shot.label });
                    console.log(`  [Shot] Generating: ${shot.label} (${shotId}) via ${provider}...`);
                    toGenerate.push(shotId);
                }
            }

            const parallelTasks = toGenerate.map(shotId => {
                const shot = shotCatalog[shotId];
                if (!shot) return Promise.resolve();
                return generateShot(shotId, refsWithAnchor, customInstruction, hasAnchor, provider, autoMatchRing, multiPiece, resolution, !resume, brandId, overlayOpts)
                    .then(result => {
                        const p = path.join(outDir, `${shotId}.png`);
                        fs.writeFileSync(p, Buffer.from(result.base64, 'base64'));
                        send({ type: 'angle_done', product: productName, angle: shotId, label: shot.label, savedTo: p, provider: result.provider });
                        send({ type: 'usage', usage: usageStats });
                        console.log(`  [Shot] ✓ ${shot.label} → ${p}`);
                    })
                    .catch(err => {
                        if (isCancelled(err) || batchCancelled) {
                            console.log(`  [Shot] ⊘ Cancelled: ${shot.label}`);
                        } else {
                            send({ type: 'angle_error', product: productName, angle: shotId, message: err.message });
                            console.error(`  [Shot] ✗ ${shot.label}: ${err.message}`);
                        }
                    });
            });

            await Promise.all(parallelTasks);

            // Per-product cost tracking: count successful shots for this product
            let successfulShots = 0;
            for (const sid of toGenerate) {
                if (fs.existsSync(path.join(outDir, `${sid}.png`))) successfulShots++;
            }
            // Include anchor if it was generated this run (not resumed from disk)
            if (anchorGenerated) successfulShots++;

            const productCostRate = getCostPerImage(provider, resolution);
            const productCost = successfulShots * productCostRate;
            if (successfulShots > 0) {
                send({ type: 'product_cost', product: productName, images: successfulShots, cost: parseFloat(productCost.toFixed(3)) });
            }
        } catch (err) {
            console.error(`[Batch] ${productName}:`, err.message);
            send({ type: 'product_error', product: productName, message: err.message });
        }

        send({ type: 'product_done', product: productName });
        console.log(`[Product] ✓ Done: ${productName}`);
    }

    // ── Concurrency driver: N products in parallel via a rolling worker pool ──
    const queue = [...productDirs];
    const workers = Array.from({ length: Math.min(PRODUCT_CONCURRENCY, queue.length) }, async () => {
        while (queue.length && !clientDisconnected && !batchCancelled) {
            const next = queue.shift();
            if (!next) break;
            await processProduct(next.name, next.fullPath);
        }
    });
    await Promise.all(workers);

    if (batchCancelled) {
        send({ type: 'cancelled', message: 'Batch cancelled by user.' });
    }

    clearInterval(heartbeat);

    const wasCancelled = batchCancelled;
    activeBatchId = null;
    batchCancelled = false;
    batchAbortController = null;
    skipProducts = new Set();

    if (!wasCancelled && !clientDisconnected) {
        send({ type: 'done' });
        console.log(`\n[Batch] Complete — $${usageStats.session.total.cost.toFixed(3)} spent, ${usageStats.session.total.images} images`);
    }
    if (!clientDisconnected) res.end();
});

// ── Native folder picker (macOS) ────────────────────────────────────────────
app.post('/pick-folder', (req, res) => {
    if (process.platform !== 'darwin') {
        return res.status(501).json({ error: 'Folder picker is only supported on macOS.' });
    }
    const script = [
        'try',
        'set f to choose folder with prompt "Select the batch folder of products"',
        'return POSIX path of f',
        'on error',
        'return ""',
        'end try',
    ];
    const args = [];
    for (const line of script) { args.push('-e', line); }
    execFile('osascript', args, { timeout: 5 * 60 * 1000 }, (err, stdout) => {
        if (err) return res.status(500).json({ error: err.message });
        const picked = (stdout || '').trim();
        if (!picked) return res.json({ cancelled: true });
        // Strip trailing slash for a cleaner path
        res.json({ path: picked.replace(/\/$/, '') });
    });
});

// ── Prompt inspector (returns the exact prompt a shot would be sent) ────────
app.post('/prompt-preview', (req, res) => {
    const { shotId, customInstruction, hasAnchor, autoMatchRing, multiPiece, brand } = req.body || {};
    const brandId = BRANDS[brand] ? brand : DEFAULT_BRAND;
    const brandShots = buildShotCatalog(brandId);
    if (!shotId || !brandShots[shotId]) {
        return res.status(400).json({ error: 'Unknown shotId.' });
    }
    const prompt = buildShotPrompt(
        shotId,
        (customInstruction || '').trim() || null,
        !!hasAnchor,
        autoMatchRing === true || autoMatchRing === '1' || autoMatchRing === 1,
        multiPiece === true || multiPiece === '1' || multiPiece === 1,
        brandId
    );
    res.json({ shotId, prompt });
});

// ── Batch retry single shot ─────────────────────────────────────────────────
app.post('/retry-angle', upload.none(), async (req, res) => {
    const { productFolder, angleId, provider: retryProvider, feedback, autoMatchRing: rawAmr, multiPiece: rawMp, resolution: rawRes, brand: rawBrand } = req.body;
    const provider = (retryProvider || 'gemini').trim();
    const resolution = ['draft', 'standard', 'high'].includes(rawRes) ? rawRes : 'standard';
    const autoMatchRing = rawAmr === '1' || rawAmr === 1 || rawAmr === true;
    const multiPiece = rawMp === '1' || rawMp === 1 || rawMp === true;
    const brandId = BRANDS[rawBrand] ? rawBrand : DEFAULT_BRAND;
    const overlayOpts = parseOverlayOpts(req.body, brandId);
    if (!productFolder || !angleId) return res.status(400).json({ error: 'Missing productFolder or angleId.' });

    const shot = buildShotCatalog(brandId)[angleId];
    if (!shot) return res.status(400).json({ error: 'Unknown shot type.' });

    const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
    const imageFiles = fs.readdirSync(productFolder)
        .filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.'))
        .map(f => {
            const full = path.join(productFolder, f);
            let size = 0;
            try { size = fs.statSync(full).size; } catch (e) {}
            return { full, size };
        })
        .sort((a, b) => b.size - a.size)
        .map(x => x.full);

    if (imageFiles.length === 0) return res.status(400).json({ error: 'No source images in product folder.' });

    // Feedback becomes a high-priority correction in the prompt's ADDITIONAL DIRECTION slot
    const feedbackText = typeof feedback === 'string' ? feedback.trim() : '';
    const correction = feedbackText
        ? `USER FEEDBACK ON PREVIOUS ATTEMPT — FIX THIS: ${feedbackText}`
        : null;

    try {
        const imageInputs = await Promise.all(imageFiles.map(async (fp) => {
            const rawFile = await fs.promises.readFile(fp);
            const buf = await toJpeg(fp, rawFile);
            return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
        }));

        const result = await generateShot(angleId, imageInputs, correction, false, provider, autoMatchRing, multiPiece, resolution, false, brandId, overlayOpts);
        const outPath = path.join(productFolder, '..', 'output', path.basename(productFolder), `${angleId}.png`);
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, Buffer.from(result.base64, 'base64'));
        res.json({ success: true, base64: result.base64, provider: result.provider, usage: usageStats });
    } catch (err) {
        console.error('[Retry Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Retry failed.' });
    }
});

// ── Retrofit overlay onto already-generated images ────────────────────────
// Client uploads finished shots + optional weight text; server returns the
// same images with the brand logo + weight composited on top. Lets users
// tweak the weight caption (or apply the overlay to older batch runs) without
// regenerating from scratch.
app.post('/apply-overlay', upload.array('images', 50), async (req, res) => {
    try {
        const brandId    = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
        const brand      = resolveBrand(brandId);
        if (!brand.overlay || !brand.overlay.supported) {
            return res.status(400).json({ error: `Brand "${brand.label}" does not support overlays.` });
        }
        const weightText = (req.body.weightText || '').trim();
        const files      = req.files || [];
        if (files.length === 0) return res.status(400).json({ error: 'No images provided.' });

        console.log(`[Overlay] brand=${brandId} applying to ${files.length} image(s) — weight: "${weightText || 'none'}"`);

        const results = [];
        for (const file of files) {
            const b64 = file.buffer.toString('base64');
            const overlaid = await applyOverlay(b64, weightText, brandId);
            const meta = await sharp(Buffer.from(overlaid, 'base64')).metadata();
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

// ── WhatsApp caption generation ────────────────────────────────────────────
app.post('/generate-caption', upload.array('images[]', 10), async (req, res) => {
    const productName   = (req.body.productName || '').trim() || 'this piece';
    const extraContext   = (req.body.extraContext || '').trim();
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
                const response = await geminiClient.models.generateContent({
                    model: 'gemini-3.1-pro-preview',
                    contents: [{ role: 'user', parts }],
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
                    contents: [{ role: 'user', parts: [{ text: captionPrompt }] }],
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
    const { images, brand: rawBrand } = req.body;
    if (!images || !Array.isArray(images) || images.length === 0) return res.status(400).json({ error: 'No images.' });

    const brandId = BRANDS[rawBrand] ? rawBrand : DEFAULT_BRAND;
    const brand = resolveBrand(brandId);

    const entries = images.map((img, i) => ({
        name: img.name || `image-${i + 1}.png`,
        data: Buffer.from(img.data, 'base64'),
    }));

    const zipBuf = buildZip(entries);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename="${brand.zipFilename}"`);
    res.send(zipBuf);
});

// ── Shopify export endpoint (#19) ─────────────────────────────────────────
app.post('/shopify/upload', async (req, res) => {
    const { productHandle, images, shopDomain, accessToken } = req.body || {};
    const domain = shopDomain || process.env.SHOPIFY_DOMAIN;
    const token = accessToken || process.env.SHOPIFY_ACCESS_TOKEN;

    if (!domain || !token) {
        return res.status(400).json({ error: 'Shopify credentials not configured' });
    }
    if (!productHandle) {
        return res.status(400).json({ error: 'Missing productHandle' });
    }
    if (!images || !Array.isArray(images) || images.length === 0) {
        return res.status(400).json({ error: 'No images provided' });
    }

    try {
        // Fetch product ID by handle
        const productRes = await fetch(
            `https://${domain}/admin/api/2024-01/products.json?handle=${encodeURIComponent(productHandle)}`,
            { headers: { 'X-Shopify-Access-Token': token, 'Content-Type': 'application/json' } }
        );
        if (!productRes.ok) {
            return res.status(productRes.status).json({ error: `Shopify API error: ${productRes.status} ${productRes.statusText}` });
        }
        const productData = await productRes.json();
        const product = productData.products?.[0];
        if (!product) {
            return res.status(404).json({ error: `Product not found with handle: ${productHandle}` });
        }

        // Upload each image
        let uploaded = 0;
        for (const img of images) {
            const imgRes = await fetch(
                `https://${domain}/admin/api/2024-01/products/${product.id}/images.json`,
                {
                    method: 'POST',
                    headers: { 'X-Shopify-Access-Token': token, 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: { attachment: img.data } }),
                }
            );
            if (imgRes.ok) uploaded++;
            else console.warn(`[Shopify] Failed to upload image ${img.name}: ${imgRes.status}`);
        }

        res.json({ success: true, uploaded });
    } catch (err) {
        console.error('[Shopify Error]', err.message);
        res.status(500).json({ error: err.message });
    }
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
async function generateShot(shotId, imageInputs, customInstruction, hasAnchor = false, provider = 'gemini', autoMatchRing = false, multiPiece = false, resolution = 'standard', noCache = false, brandId = DEFAULT_BRAND, overlayOpts = null) {
    const startedAt = Date.now();
    const prompt   = buildShotPrompt(shotId, customInstruction, hasAnchor, autoMatchRing, multiPiece, brandId);
    const isEcom   = shotId.startsWith('ecom_');
    const shotLabel = buildShotCatalog(brandId)[shotId]?.label || shotId;

    const maybeOverlay = async (b64) => {
        if (!overlayOpts || !overlayOpts.enabled) return b64;
        try {
            return await applyOverlay(b64, overlayOpts.weightText || '', brandId);
        } catch (err) {
            console.warn(`[Overlay] failed for ${shotId}: ${err.message}`);
            return b64;
        }
    };

    // 1. Cache hit? Skip the API call entirely (unless caller opts out).
    // Note: cache stores the raw post-processed image; overlay is applied on
    // the way out so toggling overlay doesn't require cache invalidation.
    const cacheKey = outputCacheKey(provider, shotId, prompt, imageInputs);
    if (!noCache) {
        const cached = outputCacheGet(cacheKey);
        if (cached) {
            console.log(`[Cache] HIT ${shotId} (${provider}) — $0`);
            audit({ shotId, provider, cache: true, durationMs: Date.now() - startedAt });
            return { base64: await maybeOverlay(cached), provider };
        }
    } else {
        console.log(`[Cache] BYPASS ${shotId} (fresh batch)`);
    }

    // 2. Enforce daily budget cap if configured
    budgetCheck();

    // One generation attempt against a specific provider
    const runOne = async (p, extraInstruction) => {
        const effectivePrompt = extraInstruction
            ? `${prompt}\n\nQC FEEDBACK FROM PREVIOUS ATTEMPT — FIX THIS: ${extraInstruction}`
            : prompt;
        let raw;
        if (p === 'openai')         raw = await generateWithOpenAI(effectivePrompt, imageInputs, resolution);
        else if (p === 'nanobana2') raw = await generateWithNanoBana2(effectivePrompt, imageInputs, resolution);
        else                        raw = await generateWithGemini(effectivePrompt, imageInputs);
        trackUsage(p, shotId, resolution);
        return raw;
    };

    // 3. Primary provider, with automatic fallback on 429/transient errors
    const chain = [provider, ...availableFallbacks(provider)];
    let raw, usedProvider, lastErr;
    for (let i = 0; i < chain.length; i++) {
        const p = chain[i];
        try {
            raw = await runOne(p);
            usedProvider = p;
            if (i > 0) console.log(`[Fallback] ✓ Succeeded on ${p} after ${chain[0]} failed.`);
            break;
        } catch (err) {
            if (isCancelled(err) || batchCancelled) throw new CancelledError();
            lastErr = err;
            if (i === chain.length - 1 || !isFallbackWorthy(err)) {
                audit({ shotId, provider, ok: false, error: err.message, durationMs: Date.now() - startedAt });
                throw err;
            }
            console.log(`[Fallback] ${p} failed (${err.message?.slice(0, 80)}), trying ${chain[i + 1]}...`);
        }
    }

    // 4. Post-process ecom shots: white-point lock + gentle sharpen
    if (isEcom) raw = await postProcessEcom(raw);

    // 5. QC pass — if score below threshold, retry ONCE with defect feedback
    const qc = await qcShot(raw, imageInputs, shotLabel);
    let qcRetried = false;
    if (!qc.skipped && qc.score < QC_THRESHOLD && qc.defects && qc.defects.toLowerCase() !== 'none') {
        console.log(`[QC] ${shotId} scored ${qc.score}/10 (< ${QC_THRESHOLD}). Retrying with feedback: ${qc.defects}`);
        qcRetried = true;
        try {
            let retryRaw = await runOne(usedProvider, qc.defects);
            if (isEcom) retryRaw = await postProcessEcom(retryRaw);
            const qc2 = await qcShot(retryRaw, imageInputs, shotLabel);
            if (qc2.skipped || qc2.score >= qc.score) {
                console.log(`[QC] retry scored ${qc2.skipped ? 'n/a' : qc2.score}, keeping retry.`);
                raw = retryRaw;
            } else {
                console.log(`[QC] retry scored worse (${qc2.score}), keeping original.`);
            }
        } catch (err) {
            console.warn(`[QC] retry generation failed, keeping original: ${err.message}`);
        }
    } else if (!qc.skipped) {
        console.log(`[QC] ${shotId} scored ${qc.score}/10 ✓`);
    }

    // 6. Cache + audit
    outputCachePut(cacheKey, raw);
    audit({
        shotId, provider: usedProvider, ok: true,
        qcScore:   qc.skipped ? null : qc.score,
        qcDefects: qc.defects || null,
        qcRetried,
        durationMs: Date.now() - startedAt,
    });
    return { base64: await maybeOverlay(raw), provider: usedProvider };
}

async function generateWithGemini(prompt, imageInputs) {
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    const raw = await callGemini(parts);
    return makeSquareBase64(raw);
}

async function generateWithOpenAI(prompt, imageInputs, resolution = 'standard') {
    const openaiResMap = { draft: '512x512', standard: '1024x1024', high: '1024x1024' };
    const openaiQualMap = { draft: 'low', standard: 'high', high: 'high' };
    const oaiSize = openaiResMap[resolution] || '1024x1024';
    const oaiQuality = openaiQualMap[resolution] || 'high';
    await acquireGeminiSlot(); // reuse the same concurrency limiter
    try {
        throwIfCancelled();
        // Use gpt-image-1.5 via the Images API with reference images
        const imageFiles = imageInputs.map((img, i) => {
            const buf = Buffer.from(img.base64, 'base64');
            return new File([buf], `ref_${i}.png`, { type: img.mimeType });
        });

        console.log(`[OpenAI] calling gpt-image-1.5... (${imageFiles.length} reference image(s), ${oaiSize} ${oaiQuality})`);

        const signal = batchAbortController?.signal;
        const response = await withCancel(withTimeout(
            openaiClient.images.edit({
                model: 'gpt-image-1.5',
                image: imageFiles,
                prompt: prompt,
                n: 1,
                size: oaiSize,
                quality: oaiQuality,
            }, signal ? { signal } : undefined),
            API_TIMEOUT_MS,
            'OpenAI'
        ));

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

async function generateWithNanoBana2(prompt, imageInputs, resolution = 'standard') {
    const nanobanaResMap = { draft: '512', standard: '1K', high: '2K' };
    const nanoImageSize = nanobanaResMap[resolution] || '1K';
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    const raw = await callNanoBana2(parts, 0, nanoImageSize);
    return makeSquareBase64(raw);
}

// ── Timeout helper ────────────────────────────────────────────────────────
const API_TIMEOUT_MS = 240000; // 4 minutes per API call (Gemini 3 Pro can be slow under load)

function withTimeout(promise, ms, label = 'API call') {
    return Promise.race([
        promise,
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`${label} timed out after ${ms / 1000}s`)), ms)
        ),
    ]);
}

// ── Shared Gemini call with retry + backoff + concurrency ───────────────────
const MAX_RETRIES = 3;
const RETRY_DELAYS = [2000, 5000, 10000];
// Separate, more generous schedule for 429/quota errors — 6 attempts over ~6 minutes
const QUOTA_MAX_RETRIES = 6;
const QUOTA_RETRY_DELAYS = [15000, 30000, 60000, 90000, 120000, 180000];

function is429(err) {
    const msg = String(err?.message || err || '');
    return /\b429\b|RESOURCE_EXHAUSTED|Too Many Requests|quota/i.test(msg);
}

function retryDelay(err, attempt) {
    if (is429(err)) return QUOTA_RETRY_DELAYS[attempt] || 180000;
    return RETRY_DELAYS[attempt] || 5000;
}

function retryCap(err) {
    return is429(err) ? QUOTA_MAX_RETRIES : MAX_RETRIES;
}

let MAX_CONCURRENT = 3;
let PRODUCT_CONCURRENCY = Math.max(1, parseInt(process.env.PRODUCT_CONCURRENCY || '2', 10));
let activeGeminiCalls = 0;
const geminiQueue = [];

function acquireGeminiSlot() {
    return new Promise((resolve, reject) => {
        if (batchCancelled) return reject(new CancelledError());
        if (activeGeminiCalls < MAX_CONCURRENT) {
            activeGeminiCalls++;
            resolve();
        } else {
            geminiQueue.push({ resolve, reject });
        }
    });
}

function releaseGeminiSlot() {
    activeGeminiCalls--;
    while (geminiQueue.length > 0) {
        const entry = geminiQueue.shift();
        if (batchCancelled) { entry.reject(new CancelledError()); continue; }
        activeGeminiCalls++;
        entry.resolve();
        break;
    }
}

async function callGemini(parts, attempt = 0) {
    await acquireGeminiSlot();
    try {
        throwIfCancelled();
        console.log(`[Gemini] calling... (${parts.filter(p => p.inlineData).length} image(s))${attempt > 0 ? ` [retry ${attempt}]` : ''}`);
        const response = await withCancel(withTimeout(
            geminiClient.models.generateContent({
                model: 'gemini-3-pro-image-preview',
                contents: [{ role: 'user', parts }],
                config: { responseModalities: ['TEXT', 'IMAGE'] },
            }),
            API_TIMEOUT_MS,
            'Gemini'
        ));

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
        if (isCancelled(err) || batchCancelled) throw new CancelledError();
        const cap = retryCap(err);
        if (attempt < cap - 1) {
            const delay = retryDelay(err, attempt);
            const label = is429(err) ? 'quota-backoff' : 'retry';
            console.log(`[Gemini] ${label} ${attempt + 1}/${cap} in ${Math.round(delay / 1000)}s (${err?.message?.slice(0, 80) || ''})`);
            await abortableSleep(delay);
            return callGemini(parts, attempt + 1);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

async function callNanoBana2(parts, attempt = 0, imageSize = '2K') {
    await acquireGeminiSlot();
    try {
        throwIfCancelled();
        console.log(`[NanoBana2] calling gemini-3.1-flash-image-preview... (${parts.filter(p => p.inlineData).length} image(s), ${imageSize})${attempt > 0 ? ` [retry ${attempt}]` : ''}`);
        const response = await withCancel(withTimeout(
            geminiClient.models.generateContent({
                model: 'gemini-3.1-flash-image-preview',
                contents: [{ role: 'user', parts }],
                config: {
                    responseModalities: ['TEXT', 'IMAGE'],
                    imageConfig: {
                        aspectRatio: '1:1',
                        imageSize: imageSize,
                    },
                },
            }),
            API_TIMEOUT_MS,
            'NanoBana2'
        ));

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
        if (isCancelled(err) || batchCancelled) throw new CancelledError();
        const cap = retryCap(err);
        if (attempt < cap - 1) {
            const delay = retryDelay(err, attempt);
            const label = is429(err) ? 'quota-backoff' : 'retry';
            console.log(`[NanoBana2] ${label} ${attempt + 1}/${cap} in ${Math.round(delay / 1000)}s (${err?.message?.slice(0, 80) || ''})`);
            await abortableSleep(delay);
            return callNanoBana2(parts, attempt + 1, imageSize);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

// ── Overlay: weight text + brand logo ──────────────────────────────────────
// Top-left weight text + top-right brand logo, scaled relative to a 3000px
// reference canvas so it reads the same at every output size. Per-brand
// logo path lives in the brand registry; if the logo file is missing we
// just render the weight text.
const OVERLAY_FONT_FAMILY = 'Futura LT';

async function applyOverlay(base64, weightText, brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    const overlayCfg = brand.overlay;
    if (!overlayCfg || !overlayCfg.supported) return base64;

    const buf = Buffer.from(base64, 'base64');
    const meta = await sharp(buf).metadata();
    const w = meta.width;
    const h = meta.height;

    const pad       = Math.round(w * (125 / 3000));
    const fontSize  = Math.round(w * (143 / 3000));
    const logoWidth = Math.round(w * (580 / 3000));

    const composites = [];

    if (weightText && weightText.trim()) {
        const textLeftPad = Math.round(w * (120 / 3000));
        const textSvg = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${fontSize * 2}">
            <text x="${textLeftPad}" y="${fontSize * 1.1}" font-family="${OVERLAY_FONT_FAMILY}" font-size="${fontSize}" font-weight="300" fill="white" letter-spacing="2">${weightText.trim()}</text>
        </svg>`);
        const textPad = Math.round(w * (100 / 3000));
        composites.push({ input: textSvg, top: textPad, left: 0 });
    }

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

    // Content-addressed JPEG cache — skip sips/sharp on repeat runs of the same bytes
    const bufHash = sha1(buffer);
    const cachedJpg = jpegCacheGet(bufHash);
    if (cachedJpg) return cachedJpg;

    if (ext === '.heic' || ext === '.heif') {
        // Try sips first (macOS native, only on darwin), then sharp as fallback
        if (process.platform === 'darwin') {
            const tmpDir = path.join(__dirname, '.tmp');
            fs.mkdirSync(tmpDir, { recursive: true });
            const stamp  = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
            const tmpIn  = path.join(tmpDir, `heic-in-${stamp}.heic`);
            const tmpOut = path.join(tmpDir, `heic-out-${stamp}.jpg`);
            try {
                fs.writeFileSync(tmpIn, buffer);
                await withTimeout(
                    new Promise((resolve, reject) => {
                        execFile('sips', ['-s', 'format', 'jpeg', tmpIn, '--out', tmpOut], err => err ? reject(err) : resolve());
                    }),
                    30000,
                    'HEIC conversion'
                );
                const out = fs.readFileSync(tmpOut);
                jpegCachePut(bufHash, out);
                return out;
            } catch (sipsErr) {
                console.warn(`[HEIC] sips failed (${sipsErr.message}), trying sharp...`);
            } finally {
                if (fs.existsSync(tmpIn)) fs.unlinkSync(tmpIn);
                if (fs.existsSync(tmpOut)) fs.unlinkSync(tmpOut);
            }
        }
        // sharp fallback (all platforms, or after sips failure on macOS)
        try {
            const out = await sharp(buffer).jpeg({ quality: 95 }).toBuffer();
            jpegCachePut(bufHash, out);
            return out;
        } catch (sharpErr) {
            console.error(`[HEIC] sharp failed (${sharpErr.message}), skipping file`);
            throw new Error(`Cannot convert HEIC file: ${filePathOrName}`);
        }
    }
    const out = await sharp(buffer).jpeg({ quality: 95 }).toBuffer();
    jpegCachePut(bufHash, out);
    return out;
}

// ── Cache eviction (#12) ──────────────────────────────────────────────────
function evictCache() {
    try {
        const files = fs.readdirSync(OUTPUT_CACHE_DIR).map(f => {
            const full = path.join(OUTPUT_CACHE_DIR, f);
            try {
                const stat = fs.statSync(full);
                return { full, size: stat.size, mtimeMs: stat.mtimeMs };
            } catch (e) { return null; }
        }).filter(Boolean);

        const now = Date.now();
        const SEVEN_DAYS = 7 * 24 * 60 * 60 * 1000;
        const MAX_CACHE_BYTES = 500 * 1024 * 1024;

        let evictedCount = 0;
        let evictedBytes = 0;

        // Pass 1: delete files older than 7 days
        const remaining = [];
        for (const f of files) {
            if (now - f.mtimeMs > SEVEN_DAYS) {
                try { fs.unlinkSync(f.full); evictedCount++; evictedBytes += f.size; } catch (e) {}
            } else {
                remaining.push(f);
            }
        }

        // Pass 2: cap total size at 500MB, delete oldest first
        remaining.sort((a, b) => a.mtimeMs - b.mtimeMs);
        let totalSize = remaining.reduce((s, f) => s + f.size, 0);
        while (totalSize > MAX_CACHE_BYTES && remaining.length > 0) {
            const oldest = remaining.shift();
            try { fs.unlinkSync(oldest.full); evictedCount++; evictedBytes += oldest.size; totalSize -= oldest.size; } catch (e) {}
        }

        if (evictedCount > 0) {
            console.log(`[CacheEvict] Evicted ${evictedCount} file(s), ${(evictedBytes / 1024 / 1024).toFixed(1)}MB freed`);
        }
    } catch (e) {
        console.warn('[CacheEvict] Error:', e.message);
    }
}

// ── Audit log rotation (#13) ─────────────────────────────────────────────
function rotateAuditLog() {
    try {
        if (!fs.existsSync(AUDIT_LOG_PATH)) return;
        const stat = fs.statSync(AUDIT_LOG_PATH);
        if (stat.size > 10 * 1024 * 1024) {
            const rotatedPath = AUDIT_LOG_PATH + '.1';
            fs.renameSync(AUDIT_LOG_PATH, rotatedPath);
            fs.writeFileSync(AUDIT_LOG_PATH, '');
            console.log(`[AuditRotate] Rotated audit.log (${(stat.size / 1024 / 1024).toFixed(1)}MB) → audit.log.1`);
        }
    } catch (e) {
        console.warn('[AuditRotate] Error:', e.message);
    }
}

// ── Persist usage stats (#14) ─────────────────────────────────────────────
function loadUsageStats() {
    try {
        if (fs.existsSync(USAGE_STATS_PATH)) {
            const saved = JSON.parse(fs.readFileSync(USAGE_STATS_PATH, 'utf8'));
            if (saved.session) Object.assign(usageStats.session, saved.session);
            if (saved.history) usageStats.history = saved.history;
            console.log('[UsageStats] Loaded from disk');
        }
    } catch (e) {
        console.warn('[UsageStats] Could not load:', e.message);
    }
}

let _usageWriteTimer = null;
function debounceSaveUsageStats() {
    if (_usageWriteTimer) return;
    _usageWriteTimer = setTimeout(() => {
        _usageWriteTimer = null;
        try {
            fs.writeFileSync(USAGE_STATS_PATH, JSON.stringify(usageStats, null, 2));
        } catch (e) { /* non-fatal */ }
    }, 5000);
}

// ── Start ───────────────────────────────────────────────────────────────────
// Run startup tasks
rotateAuditLog();
loadUsageStats();
evictCache();
setInterval(evictCache, 30 * 60 * 1000);

const server = app.listen(PORT, () => {
    const brandNames = Object.values(BRANDS).map(b => b.label).join(' + ');
    console.log(`\nUnified Pipeline (${brandNames}) → http://localhost:${PORT}\n`);
});

// Keep connections alive and prevent premature drops
server.keepAliveTimeout = 120000;      // 2 minutes
server.headersTimeout   = 125000;      // slightly above keepAliveTimeout
server.requestTimeout   = 0;          // no timeout on requests (SSE streams are long-lived)
