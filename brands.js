// Brand registry — drives copy, assets, and brand-specific shot availability.
// The HTTP server reads the active brand from req.body.brand / req.query.brand
// (falls back to DEFAULT_BRAND). All brand-specific prompt text flows from here.

const BRANDS = {
    mina: {
        id: 'mina',
        label: 'House of Mina',
        shortLabel: 'Mina',
        domain: 'houseofmina.store',
        logo: '/assets/brands/mina/logo.png',
        logoDark: '/assets/brands/mina/logo.png',
        tagline: 'luxury South Asian jewelry brand based in Karachi',
        zipFilename: 'house-of-mina-shots.zip',
        zipSelectedFilename: 'house-of-mina-selected.zip',
        scrapePlaceholder: 'https://houseofmina.store/products/...',
        // Prepended to every shot prompt — sets brand voice + fidelity rules.
        baseIntro: `You are generating product photography for House of Mina (houseofmina.store), a luxury South Asian jewelry brand. Their aesthetic is warm, elegant, and editorial — rich gold tones, deep jewel colors, and a regal yet modern sensibility.

Copy the jewelry from the reference photo(s) with absolute fidelity. Reproduce every stone, every metal tone, every proportion, every surface texture exactly. Do not add, remove, merge, or alter any design element. The generated image must be indistinguishable from a real photograph of this exact piece.

PHYSICAL SCALE IS SACRED: the reference photo represents the piece at its TRUE real-world size. When the piece is shown against a body (wrist, hand, neck, ear, finger) or a prop, it MUST maintain realistic proportions against adult human anatomy. Do NOT enlarge the piece "to make it look impressive," do NOT shrink it "to fit the composition." A bangle fits barely over the knuckles. A pendant is 1–3 cm across, not a medallion. A ring covers one knuckle segment. If the reference already shows the piece worn, measure it against the anatomy in the reference and reproduce that exact ratio.`,
        // How the brand name appears in the ecom_stand scene.
        ecomStandBrandRef: 'House of Mina brand display presentation',
        // WhatsApp caption copywriter system prompt — brand voice comes from here.
        captionSystem: `You are the copywriter for House of Mina (houseofmina.store), a luxury South Asian jewelry brand based in Karachi. You write WhatsApp community posts to showcase new jewelry pieces.`,
        captionBrandMention: '*House of Mina*',
        // Extra brand-only shot types injected into the shot catalog.
        extraShots: [],
        // Logo+weight overlay composited onto finished ecom shots. Mina has no
        // overlay logo yet, so overlay is opt-in but will render weight-text only.
        overlay: {
            supported: true,
            logoPath: 'public/assets/brands/mina/logo.png',
            defaultEnabled: false,
        },
    },

    taheri: {
        id: 'taheri',
        label: 'Taheri Collections',
        shortLabel: 'Taheri',
        domain: 'tahericollections.com',
        logo: '/assets/brands/taheri/logo.png',
        logoDark: '/assets/brands/taheri/logo-dark.png',
        tagline: 'Taheri Collections — signature jewelry with an editorial, heritage-modern sensibility',
        zipFilename: 'taheri-shots.zip',
        zipSelectedFilename: 'taheri-selected.zip',
        scrapePlaceholder: 'https://tahericollections.com/products/...',
        baseIntro: `You are generating product photography for Taheri Collections, a signature jewelry house with an editorial, heritage-modern sensibility — warm gold tones, deep jewel colors, and a regal yet contemporary voice.

Copy the jewelry from the reference photo(s) with absolute fidelity. Reproduce every stone, every metal tone, every proportion, every surface texture exactly. Do not add, remove, merge, or alter any design element. The generated image must be indistinguishable from a real photograph of this exact piece.

PHYSICAL SCALE IS SACRED: the reference photo represents the piece at its TRUE real-world size. When the piece is shown against a body (wrist, hand, neck, ear, finger) or a prop, it MUST maintain realistic proportions against adult human anatomy. Do NOT enlarge the piece "to make it look impressive," do NOT shrink it "to fit the composition." A bangle fits barely over the knuckles. A pendant is 1–3 cm across, not a medallion. A ring covers one knuckle segment. If the reference already shows the piece worn, measure it against the anatomy in the reference and reproduce that exact ratio.`,
        ecomStandBrandRef: 'Taheri Collections brand display presentation',
        captionSystem: `You are the copywriter for Taheri Collections, a signature jewelry house. You write WhatsApp community posts to showcase new jewelry pieces.`,
        captionBrandMention: '*Taheri Collections*',
        // Taheri's signature deliverable — logo top-right, weight-text top-left,
        // scaled against a 3000px reference so it looks right at any output size.
        overlay: {
            supported: true,
            logoPath: 'public/assets/brands/taheri/logo-dark.png',
            defaultEnabled: true,
        },
        extraShots: [
            {
                id: 'taheri_signature',
                label: 'Taheri Signature',
                category: 'taheri',
                description: 'Wooden stand/bust on emerald green — auto-detects jewelry type',
                scenePrompt: `SCENE: Taheri brand signature product photography. Study the reference photo(s) VERY carefully to determine the jewelry type and number of pieces, then follow the exact display rules below.

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
            },
        ],
    },
};

const DEFAULT_BRAND = 'mina';

function resolveBrand(id) {
    if (id && BRANDS[id]) return BRANDS[id];
    return BRANDS[DEFAULT_BRAND];
}

function listBrands() {
    return Object.values(BRANDS).map(b => ({
        id: b.id,
        label: b.label,
        shortLabel: b.shortLabel,
        domain: b.domain,
        logo: b.logo,
        logoDark: b.logoDark,
        scrapePlaceholder: b.scrapePlaceholder,
        overlay: b.overlay
            ? { supported: !!b.overlay.supported, defaultEnabled: !!b.overlay.defaultEnabled }
            : { supported: false, defaultEnabled: false },
    }));
}

module.exports = { BRANDS, DEFAULT_BRAND, resolveBrand, listBrands };
