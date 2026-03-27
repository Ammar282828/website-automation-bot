require('dotenv').config();

const express      = require('express');
const multer       = require('multer');
const path         = require('path');
const fs           = require('fs');
const os           = require('os');
const { execFile } = require('child_process');
const sharp        = require('sharp');
const { GoogleGenAI } = require('@google/genai');

// Prevent unhandled errors from crashing the server
process.on('uncaughtException', (err) => {
    console.error('[Uncaught Exception]', err.message, err.stack);
});
process.on('unhandledRejection', (reason) => {
    console.error('[Unhandled Rejection]', reason);
});

if (!process.env.GOOGLE_API_KEY) {
    console.error('ERROR: GOOGLE_API_KEY is not set in your .env file.');
    process.exit(1);
}

const PORT = process.env.PORT || 3000;
const ai   = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });

const ANGLES = [
    {
        id: 'front',
        label: 'Front View',
        instruction: 'Straight-on front view. Product faces directly at camera, centered. Full product visible.',
    },
    {
        id: 'elevated',
        label: '3/4 Elevated',
        instruction: '3/4 elevated angle.',
    },
    {
        id: 'band',
        label: 'Band Focus',
        instruction: 'Band/shank focus shot.',
    },
    {
        id: 'detail',
        label: 'Detail Close-up',
        instruction: 'Extreme macro close-up.',
    },
];

const app    = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

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

// ── Single / multi-image generation endpoint ────────────────────────────────
app.post('/generate', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const outputType        = req.body.outputType || 'ecommerce';
    const customInstruction = (req.body.customInstruction || '').trim() || null;

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    const primary = imageInputs[0];

    try {
        const results = {};

        if (outputType === 'ecommerce' || outputType === 'both') {
            // Generate front first, then pass it as reference to other angles for consistency
            const frontData = await generateEcommerceShot(imageInputs, customInstruction, ANGLES[0]);
            const frontRef  = { base64: frontData, mimeType: 'image/png' };

            const [elevatedData, bandData, detailData] = await Promise.all([
                generateEcommerceShot([...imageInputs, frontRef], customInstruction, ANGLES[1]),
                generateEcommerceShot([...imageInputs, frontRef], customInstruction, ANGLES[2]),
                generateEcommerceShot([...imageInputs, frontRef], customInstruction, ANGLES[3]),
            ]);

            results.ecommerce = [
                { id: ANGLES[0].id, label: ANGLES[0].label, data: frontData    },
                { id: ANGLES[1].id, label: ANGLES[1].label, data: elevatedData },
                { id: ANGLES[2].id, label: ANGLES[2].label, data: bandData     },
                { id: ANGLES[3].id, label: ANGLES[3].label, data: detailData   },
            ];
        }
        if (outputType === 'model' || outputType === 'both') {
            const raw = await generateModelShot(imageInputs, customInstruction);
            results.model = raw;
        }

        res.json({ success: true, results });
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

// ── Retry single angle ──────────────────────────────────────────────────────
app.post('/generate-angle', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const angleId           = req.body.angleId;
    const customInstruction = (req.body.customInstruction || '').trim() || null;

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        // Model shot retry
        if (angleId === 'model') {
            const imageData = await generateModelShot(imageInputs, customInstruction);
            return res.json({ success: true, imageData });
        }

        const angle = ANGLES.find(a => a.id === angleId);
        if (!angle) return res.status(400).json({ error: 'Unknown angle.' });

        const frontRefBase64 = (req.body.frontRef || '').trim().replace(/^data:image\/\w+;base64,/, '');
        const refsForAngle = angleId !== 'front' && frontRefBase64
            ? [...imageInputs, { base64: frontRefBase64, mimeType: 'image/png' }]
            : imageInputs;

        const imageData = await generateEcommerceShot(refsForAngle, customInstruction, angle);
        res.json({ success: true, imageData });
    } catch (err) {
        console.error('[Generate-Angle Error]', err?.message || err);
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

    if (!folderPath) { send({ type: 'error', message: 'No folder path provided.' }); return res.end(); }
    if (!fs.existsSync(folderPath)) { send({ type: 'error', message: `Folder not found: ${folderPath}` }); return res.end(); }
    if (!fs.statSync(folderPath).isDirectory()) { send({ type: 'error', message: 'That path is a file, not a folder. Please provide a folder containing product subfolders.' }); return res.end(); }

    const productDirs = fs.readdirSync(folderPath, { withFileTypes: true })
        .filter(d => d.isDirectory() && !d.name.startsWith('.') && d.name !== 'ecommerce' && d.name !== 'output')
        .map(d => ({ name: d.name, fullPath: path.join(folderPath, d.name) }));

    if (productDirs.length === 0) {
        send({ type: 'error', message: 'No product subfolders found.' });
        return res.end();
    }

    activeBatchId = Date.now().toString();
    batchCancelled = false;

    send({ type: 'start', total: productDirs.length, batchId: activeBatchId });

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

            // ── Generate front first (serial) ──
            let generatedFront = null;
            send({ type: 'angle_start', product: productName, angle: 'front', label: ANGLES[0].label });
            try {
                const frontBase64 = await generateEcommerceShot(imageInputs, customInstruction, ANGLES[0]);
                generatedFront = frontBase64;
                const outPath = path.join(outDir, 'front.png');
                fs.writeFileSync(outPath, Buffer.from(frontBase64, 'base64'));
                send({ type: 'angle_done', product: productName, angle: 'front', label: ANGLES[0].label, savedTo: outPath });
            } catch (err) {
                send({ type: 'angle_error', product: productName, angle: 'front', message: err.message });
            }

            if (batchCancelled) {
                send({ type: 'product_done', product: productName });
                send({ type: 'cancelled', message: 'Batch cancelled by user.' });
                break;
            }

            // ── Generate remaining angles + model in parallel (front ref for consistency) ──
            const refsForAngles = generatedFront
                ? [...imageInputs, { base64: generatedFront, mimeType: 'image/png' }]
                : imageInputs;

            for (const a of ANGLES.slice(1)) {
                send({ type: 'angle_start', product: productName, angle: a.id, label: a.label });
            }
            send({ type: 'model_start', product: productName });

            const parallelTasks = [
                generateEcommerceShot(refsForAngles, customInstruction, ANGLES[1])
                    .then(b64 => {
                        const p = path.join(outDir, 'elevated.png');
                        fs.writeFileSync(p, Buffer.from(b64, 'base64'));
                        send({ type: 'angle_done', product: productName, angle: 'elevated', label: ANGLES[1].label, savedTo: p });
                    })
                    .catch(err => send({ type: 'angle_error', product: productName, angle: 'elevated', message: err.message })),
                generateEcommerceShot(refsForAngles, customInstruction, ANGLES[2])
                    .then(b64 => {
                        const p = path.join(outDir, 'band.png');
                        fs.writeFileSync(p, Buffer.from(b64, 'base64'));
                        send({ type: 'angle_done', product: productName, angle: 'band', label: ANGLES[2].label, savedTo: p });
                    })
                    .catch(err => send({ type: 'angle_error', product: productName, angle: 'band', message: err.message })),
                generateEcommerceShot(refsForAngles, customInstruction, ANGLES[3])
                    .then(b64 => {
                        const p = path.join(outDir, 'detail.png');
                        fs.writeFileSync(p, Buffer.from(b64, 'base64'));
                        send({ type: 'angle_done', product: productName, angle: 'detail', label: ANGLES[3].label, savedTo: p });
                    })
                    .catch(err => send({ type: 'angle_error', product: productName, angle: 'detail', message: err.message })),
                generateModelShot(imageInputs, customInstruction)
                    .then(modelB64 => {
                        const p = path.join(outDir, 'model.png');
                        fs.writeFileSync(p, Buffer.from(modelB64, 'base64'));
                        send({ type: 'model_done', product: productName, savedTo: p });
                    })
                    .catch(err => send({ type: 'model_error', product: productName, message: err.message })),
            ];

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

// ── Batch retry single angle ────────────────────────────────────────────────
app.post('/retry-angle', upload.none(), async (req, res) => {
    const { productFolder, angleId } = req.body;
    if (!productFolder || !angleId) return res.status(400).json({ error: 'Missing productFolder or angleId.' });

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

        if (angleId === 'model') {
            const raw = await generateModelShot(imageInputs, null);
            const outPath = path.join(productFolder, '..', 'output', path.basename(productFolder), 'model.png');
            fs.mkdirSync(path.dirname(outPath), { recursive: true });
            fs.writeFileSync(outPath, Buffer.from(raw, 'base64'));
            return res.json({ success: true, base64: raw });
        }

        const angle = ANGLES.find(a => a.id === angleId);
        if (!angle) return res.status(400).json({ error: 'Unknown angle.' });

        const frontPath = path.join(productFolder, '..', 'output', path.basename(productFolder), 'front.png');
        const refsForAngle = angleId !== 'front' && fs.existsSync(frontPath)
            ? [...imageInputs, { base64: fs.readFileSync(frontPath).toString('base64'), mimeType: 'image/png' }]
            : imageInputs;

        const imgBase64 = await generateEcommerceShot(refsForAngle, null, angle);
        const outPath = path.join(productFolder, '..', 'output', path.basename(productFolder), `${angleId}.png`);
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, Buffer.from(imgBase64, 'base64'));
        res.json({ success: true, base64: imgBase64 });
    } catch (err) {
        console.error('[Retry Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Retry failed.' });
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
    res.setHeader('Content-Disposition', 'attachment; filename="jewelry-shots.zip"');
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

// ── Ecommerce shot ──────────────────────────────────────────────────────────
async function generateEcommerceShot(imageInputs, customInstruction, angle = ANGLES[0], hasFrontRef = null) {
    const isDetail   = angle.id === 'detail';
    const isElevated = angle.id === 'elevated';
    const isBand     = angle.id === 'band';

    const promptInputs = imageInputs;
    if (hasFrontRef === null) {
        hasFrontRef = angle.id !== 'front'
            && promptInputs.length > 1
            && promptInputs[promptInputs.length - 1]?.mimeType === 'image/png';
    }
    const numOriginalRefs = hasFrontRef ? promptInputs.length - 1 : promptInputs.length;
    const contextNote = hasFrontRef
        ? `You have ${promptInputs.length} images. The first ${numOriginalRefs} are original reference photo(s) of the jewelry \u2014 these may be DIFFERENT ANGLES of the same piece (front, side, top, close-up, etc.). Study EVERY reference image carefully: each angle reveals details that other angles may hide (e.g., a side view shows band profile, a top view shows stone arrangement, a close-up shows prong details). Build a COMPLETE mental model of the piece by combining information from ALL angles before generating. The LAST image is the APPROVED RENDERED FRONT VIEW of this exact piece \u2014 match its design exactly.`
        : numOriginalRefs > 1
            ? `You have ${numOriginalRefs} reference photos of the jewelry piece. These are DIFFERENT ANGLES of the SAME piece \u2014 front, side, top, close-up, etc. BEFORE generating anything, study EVERY single reference image and combine the information: each angle reveals details hidden in the others. A side view shows the band profile and gallery. A top view shows stone arrangement. A close-up shows prong count and texture. Build a COMPLETE mental model of the piece from ALL angles, then generate.`
            : 'You have been given one reference photo of the jewelry piece. Study every detail carefully.';

    const sceneBlock = isDetail ? [
        'SCENE \u2014 MACRO CLOSE-UP:',
        '- THIS IS AN EXTREME MAGNIFICATION SHOT. Do NOT show the full product.',
        '- Camera positioned 1\u20132 cm from the surface, filling the entire frame with a single small region.',
        '- Default subject: the primary center stone and its immediate setting. If there is no center stone, focus on the most intricate area (detailed setting, engraving, or surface texture).',
        '- The chosen fragment should fill at least 80% of the frame \u2014 crop aggressively.',
        '- Razor-sharp focus on the closest surface; gentle natural bokeh softens the background.',
        '- Jewelry rests on clean white surface; slight micro-shadow beneath the piece.',
        '- No full-product silhouette visible \u2014 this is not a product overview shot.',
        '- Square 1:1 frame.',
    ] : isElevated ? [
        'SCENE \u2014 3/4 ELEVATED:',
        '- IMPORTANT: This is the SAME ring from the front view, now shown from a 3/4 angle. Do NOT create a different ring.',
        '- Ring stands UPRIGHT on its shank on a clean white surface \u2014 NOT lying flat.',
        '- Camera is at roughly the same height as the ring (near table level), angled about 30\u201340 degrees to the side.',
        '- This is the classic jewelry-store display angle: you see the stone face AND the side profile of the setting simultaneously.',
        '- ORIGINAL REFERENCES OVERRIDE: For the basket walls, gallery, and shoulder junction, trust the original reference photo(s) over any rendered front view. Those side-geometry details must come from the real photos, not from generic ring priors.',
        '- The side of the setting, prongs, basket walls, gallery, and upper band are clearly visible \u2014 this shot reveals the 3D architecture that a front view hides.',
        '- BASKET FIDELITY: Reproduce the exact basket/setting side profile from the reference — its height, wall angle, any side decorations or cut-outs. Do NOT simplify or round off the basket.',
        '- SHOULDER FIDELITY: Reproduce the exact shoulder-to-basket junction from the reference — the same curve, step, taper break, undercut, or decorative sweep. Do NOT replace it with a generic cathedral shoulder or smooth taper.',
        '- DO NOT shoot from above. The camera must be near the ring\'s eye-level, NOT looking down at it.',
        '- Every design detail from the front view (stone count, band pattern, basket shape, setting type, metal color) MUST be visible and identical.',
        '- Soft, natural drop shadow directly beneath the piece.',
        '- Background fades to pure white at the edges.',
        '- No reflections, no gradients, no artificial glow.',
        '- Square 1:1 frame. White fill any empty areas.',
    ] : isBand ? [
        'SCENE \u2014 BAND & BASKET SIDE PROFILE:',
        hasFrontRef
            ? `- REFERENCE USAGE: You have ${numOriginalRefs} ORIGINAL reference photo(s) plus one approved rendered front view (the last image). Use the ORIGINAL reference photo(s) as the primary authority for the shoulder shape, band profile, basket structure, and gallery geometry. Use the rendered front view as an additional consistency reference for the approved front-facing design, stone layout, metal color, and finish.`
            : `- REFERENCE USAGE: Use the ${numOriginalRefs} ORIGINAL reference photo(s) as the authority for the shoulder shape, band profile, basket structure, and gallery geometry in this shot.`,
        '- The ring stands UPRIGHT on its shank, positioned so the camera sees the SIDE PROFILE of the band AND basket.',
        '- Specifically: rotate the ring so the camera is looking at the LEFT (or OUTER-LEFT) edge of the band shank.',
        '- Camera is at TABLE-SURFACE LEVEL (0\u20135 degrees elevation), looking HORIZONTALLY at the SIDE of the ring.',
        '- BASKET FIDELITY: The basket/setting side wall is clearly visible in this shot. Reproduce its EXACT profile from the reference: its height, the angle of its walls, any decorative cut-outs, claws, or architectural details on the side. Do NOT simplify or invent the basket shape.',
        '- BAND FIDELITY: Reproduce the exact band profile: width, thickness, taper, shank shape (flat, rounded, knife-edge, etc.), and any surface details (milgrain, channel stones, engravings) visible on the side face.',
        '- SHOULDER JUNCTION (CRITICAL): The exact point where the shank meets the base of the basket is UNIQUE to this ring. In the reference image(s), locate both shoulders and study their shape precisely. Reproduce them identically — the curve, the angle, any step or undercut, any decorative sweep. Do NOT invent, smooth, or generalize.',
        '- NEGATIVE CONSTRAINT: Do NOT default to a generic cathedral shoulder, donut gallery, peg-head, tulip basket, cone basket, or smooth solitaire taper unless the reference explicitly shows that exact structure.',
        '- The band and lower basket fill the frame. The stone appears at the TOP partially visible but is NOT the focus.',
        '- CRITICAL: Do NOT show the INTERIOR or UNDERSIDE of the basket — only the EXTERIOR SIDE WALL is visible from this angle.',
        '- STRICT: Do NOT invent decorative elements not present in the reference image.',
        '- Clean white surface with soft micro-shadow beneath the piece.',
        '- No reflections, no gradients.',
        '- Square 1:1 frame. White fill any empty areas.',
    ] : [
        'SCENE \u2014 TOP-DOWN 45\u00b0:',
        '- The ring lies FLAT on a clean white surface with the FRONT of the ring facing the camera \u2014 the stone/setting is the focal point, fully visible from above.',
        '- Camera is positioned above and slightly in front, at roughly 45 degrees from overhead, angled to look down at the FRONT FACE of the ring.',
        '- The stone face, prongs, and setting must be clearly visible and dominate the frame \u2014 this is a top-down view of the FRONT of the ring, not the back.',
        '- DO NOT show the back or underside of the ring. The camera must see the same front face as the standard front view, just from a higher angle.',
        '- The band should be visible curving away from the camera, providing context but not dominating.',
        '- Clean, minimal e-commerce shot \u2014 soft, natural drop shadow directly beneath the piece.',
        '- No reflections, no gradients, no artificial glow.',
        '- Square 1:1 frame. White fill any empty areas.',
    ];

    const lightingBlock = isDetail ? [
        'LIGHTING \u2014 MACRO:',
        '- Single narrow spotlight or ring flash aimed directly at the featured area',
        '- Every facet, prong, and micro-texture must be crisply lit',
        '- Diamonds: intense prismatic fire and sharp sparkle points. Gold: warm micro-reflections. Silver: cool crisp glint.',
        '- No fill lights \u2014 hard light that reveals micro-detail',
    ] : [
        'LIGHTING:',
        '- Single overhead softbox \u2014 bright but natural, not clinical',
        '- Clean specular highlights on metal and gemstones showing their exact material properties',
        '- Diamonds: sharp prismatic sparkle. Gold: warm reflection. Silver: cool crisp gleam.',
        '- No fill lights, no rim lights \u2014 one source only',
    ];

    const prompt = isBand ? [
        `You have ${numOriginalRefs} original reference photo(s) of a jewelry piece${hasFrontRef ? ', plus one approved rendered front view (the last image). The original reference photo(s) are the primary authority for every physical detail. The rendered front view is a secondary consistency check for stone layout, metal color, and finish.' : '. The original reference photo(s) are the primary authority for every physical detail.'}`,
        '',
        contextNote,
        '',
        'TASK: Generate a professional luxury product photograph of this EXACT jewelry piece for a high-end e-commerce listing. Extract the piece from its current background and place it in a controlled studio environment.',
        '',
        'SUBJECT — ABSOLUTE FIDELITY TO REFERENCE:',
        'Study every physical detail in the reference image(s) and reproduce them with zero deviation. This means:',
        '- Count the exact number of prongs and match that count precisely. Cross-check by studying the basket structure — prongs connect to the basket, confirming their count and arrangement.',
        '- Reproduce the basket exactly: height, wall thickness, side profile, any cut-outs, milgrain, or architectural details.',
        '- Reproduce the shoulder junction (where the shank meets the base of the basket) exactly as it appears in the reference — its curve, angle, any step, undercut, or decorative sweep. Study both shoulders across all reference images.',
        '- Reproduce the band profile exactly: width, thickness, taper, shank shape (flat, rounded, knife-edge), and any surface details (milgrain, channel stones, engravings).',
        '- Preserve the exact metal color, texture, gemstone shape, cut, color, and setting type.',
        '- If any detail is not clearly visible in the reference, leave it hidden or show only what the reference supports. Stay conservative with ambiguous areas — preserve the visible silhouette rather than guessing.',
        '',
        'SCENE — SIDE PROFILE VIEW:',
        'The ring stands upright on its shank. The camera looks at the left (outer-left) edge of the band, capturing the side profile of both the band and the basket.',
        '- Camera height: table-surface level, 0–5 degrees elevation, looking horizontally at the side of the ring.',
        '- The band and lower basket fill the frame. The stone appears at the top, partially visible, but is secondary to the band and setting structure.',
        '- Only the exterior side wall of the basket is visible from this angle.',
        '- Clean white (#FFFFFF) surface with a soft, natural micro-shadow beneath the piece.',
        '- Square 1:1 frame. The piece occupies roughly 65% of the frame, centered, with equal breathing room on all sides. Fill empty areas with white.',
        '',
        'LIGHTING:',
        'Single overhead softbox producing bright but natural light. Clean specular highlights that reveal exact material properties — sharp prismatic sparkle on diamonds, warm reflection on gold, cool crisp gleam on silver. Subtle, realistic shadow and reflection beneath the piece. Professional DSLR macro lens quality: extremely sharp focus across the entire piece, high resolution, luxury commercial photography.',
        '',
        'CONSTRAINTS:',
        '- Reproduce only what exists in the reference. Add nothing: no extra stones, no split shanks, no decorative elements, no design improvements.',
        '- Remove nothing: preserve every element from the reference without simplifying or merging.',
        '- Do not default to generic ring archetypes (cathedral shoulder, donut gallery, peg-head, tulip basket, cone basket, smooth solitaire taper) unless the reference explicitly shows that structure.',
        ...(customInstruction ? ['', `CUSTOM SCENE OVERRIDE: ${customInstruction}`] : []),
    ].join('\n') : [
        'Use the uploaded image(s) as the EXACT reference of the jewelry piece.',
        contextNote,
        '',
        'Extract the jewelry from whatever background or hand is in the reference and generate a professional luxury product photoshoot of the EXACT SAME piece.',
        'The jewelry must remain 100% identical to the original image(s) — do NOT change the design, shape, gemstones, metal color, texture, proportions, prong count, prong style, setting type, shank profile, or ANY details whatsoever.',
        'Do NOT add features not in the reference: no extra stones, no split shanks unless the reference has one, no decorative elements, no design "improvements."',
        'Do NOT remove, simplify, or merge any element from the reference.',
        '',
        'ZERO-TOLERANCE GEOMETRY CHECK FOR RINGS:',
        '- Basket/setting shape and the shoulder junction are LOCKED physical geometry, not stylistic interpretation.',
        '- Use the ORIGINAL reference photo(s) as the authority for basket height, basket wall angle, gallery architecture, and the exact point where the shank meets the basket.',
        '- Do NOT replace those areas with a generic ring archetype such as a cathedral shoulder, smooth taper, donut gallery, cone basket, peg-head, or tulip setting.',
        '- If the chosen camera angle would naturally hide part of the basket or shoulder, keep that area hidden or only as visible as the real reference supports. Do NOT reveal invented side geometry.',
        '- If any basket or shoulder detail is ambiguous, stay conservative and preserve the visible silhouette from the reference instead of guessing.',
        '',
        'PRONG COUNT: Count the EXACT number of prongs in the reference image. Then cross-check by studying the basket and gallery structure — the prongs connect to the basket, so the basket shape confirms the prong count and arrangement. Reproduce that EXACT number. Do NOT default to 4 prongs — if the reference has 6, 8, 12, or any other count, match it precisely. The prong count is a fixed physical property of the ring.',
        'BASKET/SETTING: Reproduce the basket exactly — its height, wall thickness, side profile shape, any decorative cut-outs, milgrain, or architectural details. The basket is as much a design element as the stone. Do NOT simplify it into a plain cone or cylinder.',
        'SHOULDER (where band meets basket): HIGHEST PRIORITY. The shoulder is the exact point where the shank widens or transitions into the base of the setting. Look at the reference and find this junction on BOTH sides of the ring. It may have a specific curve, step, undercut, cut-out, swept wing, or decorative shape. You MUST reproduce it exactly — do NOT smooth it, simplify it, or replace it with a generic taper. If you cannot see both shoulders clearly in a single reference image, look at ALL reference images provided and piece together the full picture. Inventing or guessing the shoulder shape is not acceptable.',
        '',
        ...sceneBlock,
        '',
        'Place the jewelry alone on a clean, minimal pure white (#FFFFFF) background.',
        '',
        ...lightingBlock,
        '- Subtle, realistic shadows and reflections for a high-end jewelry product shoot look.',
        '- The image should appear as if photographed using a professional DSLR with a macro lens: extremely sharp focus across the entire piece, high resolution, luxury commercial photography quality.',
        '',
        'WHITESPACE: The piece should occupy roughly 65% of the frame, centered, with equal breathing room on all sides.',
        '',
        'Do NOT modify the jewelry in ANY way. Only improve the presentation, lighting, and background.',
        'The reference image is the ONLY source of truth. If a detail is not visible in the reference, do NOT invent it.',
        '',
        'Square 1:1 output.',
        ...(customInstruction ? ['', `CUSTOM SCENE OVERRIDE: ${customInstruction}`] : []),
    ].join('\n');

    const parts = [
        { text: prompt },
        ...promptInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];

    const raw = await callGemini(parts);
    return makeSquareBase64(raw);
}

// ── Model shot ──────────────────────────────────────────────────────────────
async function generateModelShot(imageInputs, customInstruction) {
    const sceneInstruction = customInstruction
        ? `Place the jewelry in this scene: ${customInstruction}.`
        : 'Show the jewelry worn on a woman\'s hand \u2014 tight close-up cropped to ONLY the hand and wrist. No face, no body, no neck, no full arm. Just the hand. Natural, elegant hand with relaxed fingers in a graceful pose. Single soft key light from camera-left. Shallow depth of field with the jewelry in razor-sharp focus and the background gently blurred. Square 1:1 crop.';

    const prompt = [
        'You are simulating a photograph taken by a professional jewelry photographer. You have been given one reference photo of the jewelry piece.',
        '',
        'CRITICAL \u2014 reproduce the jewelry with absolute fidelity:',
        '- Every gemstone: exact color, cut style, facet count, number of stones, their arrangement and size ratios',
        '- Metal: exact color and finish (yellow gold, rose gold, silver, oxidised, brushed, polished, matte)',
        '- Every design detail: prong count, setting style, engraving, filigree, milgrain, links, clasps, chain pattern',
        '- Proportions and scale must match the reference exactly \u2014 do not resize, idealise, or simplify any element',
        '- Do NOT add stones that are not in the reference. Do NOT remove or merge design elements. Do NOT change the metal color.',
        '- IF THE JEWELRY IS A RING: preserve the exact basket/setting profile and the exact shoulder junction where the shank meets the basket. Do NOT replace them with a generic cathedral shoulder, smooth taper, cone basket, donut gallery, peg-head, or tulip setting.',
        '- IF THE JEWELRY IS A RING: if the hand pose or camera angle partly hides the basket or shoulder, keep those areas hidden rather than inventing geometry that is not supported by the reference.',
        '',
        'PHOTOGRAPHIC REALISM \u2014 this must be indistinguishable from an editorial photo in Vogue or Harper\'s Bazaar:',
        '',
        'ANTI-AI CHECKLIST (every point is mandatory):',
        '- FRAMING: Show ONLY the hand and wrist. No face, no neck, no shoulders, no full arm. Crop tightly.',
        '- Hands: visible knuckle creases, slightly uneven nail lengths, natural skin tone variation across fingers. Subtle vein texture on the back of the hand. Realistic nail beds with natural cuticles.',
        '- Skin: real skin on a woman in her early 20s \u2014 fine pore texture on the fingers and back of hand, smooth healthy look, natural tonal variation between knuckles and palm side. NO porcelain-smooth AI skin. ABSOLUTELY NO signs of aging \u2014 no wrinkles, no visible bulging veins, no sun damage, no aged hands.',
        '- Fingers: natural finger proportions, realistic joint bends, fingertips with visible fingerprint texture. Nails should have a natural manicure (not glossy gel, not bare bitten nails).',
        '',
        'LIGHTING:',
        '- One dominant key light source with clear directionality (window light from camera-left, or a single softbox above-right)',
        '- The shadow side of the hand should be noticeably darker, not filled in evenly',
        '- Jewelry should have ONE bright specular highlight and natural shadow falloff \u2014 not glowing from all directions',
        '- Avoid flat, shadowless, "product listing" lighting',
        '',
        'COMPOSITION:',
        '- Shot on an 85mm f/1.4 lens',
        '- Frame it like a real photographer would: rule of thirds, slight negative space, the jewelry at a natural visual anchor point',
        '- Slight depth compression typical of a telephoto portrait lens \u2014 background elements slightly enlarged relative to subject',
        '',
        'BACKGROUND \u2014 STUDIO:',
        '- Professional photography studio with a seamless paper or muslin backdrop in a warm neutral tone (soft grey, warm taupe, or muted beige).',
        '- The backdrop must show subtle real-world imperfections: very faint creases or wrinkles in the paper/fabric, slight tonal unevenness where the light falls off toward the edges.',
        '- Light falloff: center slightly brighter from the key light, gentle natural darkening toward the corners. NOT a perfectly uniform flat tone.',
        '',
        'COLOR:',
        '- Warm, slightly desaturated tones as if shot on Kodak Portra 400 film \u2014 soft contrast, creamy highlights, natural shadow rolloff',
        '- Skin tones should lean warm and natural, never orange or pink-shifted',
        '- Avoid over-saturation \u2014 real editorial photos are usually more muted than you\'d expect',
        '',
        'CAMERA ARTIFACTS (these make it look REAL \u2014 do not skip):',
        '- Fine film grain or sensor noise visible across the entire image, especially in shadow areas and the backdrop. This is the single most important anti-AI signal.',
        '- Very subtle chromatic aberration (color fringing) at high-contrast edges like metal against backdrop.',
        '- Natural vignetting: corners of the frame slightly darker than center.',
        '- Micro-motion: the tiniest sense of life \u2014 not everything frozen perfectly sharp, as if shot at 1/200s.',
        '- Focus falloff should feel optical (gradual, with bokeh circles on specular highlights) not computational (uniform gaussian blur).',
        '',
        'WHAT TO AVOID (common AI tells):',
        '- Perfectly noise-free, grain-free image \u2014 this screams AI. Real cameras always have sensor noise.',
        '- Perfectly smooth skin on the hand \u2014 real hands have texture',
        '- Symmetrical studio lighting with no shadow',
        '- Hyper-sharp everything \u2014 real photos have a focal plane; things before and after it go soft',
        '- Showing any part of the body beyond the hand and wrist',
        '- Jewelry that glows or emits light rather than reflecting it',
        '',
        'JEWELRY LIGHT BEHAVIOR:',
        '- The jewelry must reflect light naturally from the single key source only \u2014 no omnidirectional glow, no self-illumination, no HDR bloom on the metal',
        '- Diamonds: sharp prismatic fire from the key light. Gold: warm single-source reflection. Silver: cool crisp glint.',
        '',
        sceneInstruction,
        '',
        'The jewelry must be the absolute focal point. Every surface facet and metal texture must be visible and physically correct.',
    ].join('\n');

    const refNote = imageInputs.length > 1
        ? `You have ${imageInputs.length} reference photos of the jewelry piece from different angles. Study ALL of them to build a complete understanding of the piece before generating.`
        : 'You have been given one reference photo of the jewelry piece.';

    const parts = [
        { text: refNote + '\n\n' + prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    const raw = await callGemini(parts);
    return makeSquareBase64(raw);
}

// ── Shared Gemini call with retry + backoff + concurrency ───────────────────
const MAX_RETRIES = 3;
const RETRY_DELAYS = [2000, 5000, 10000];

// Concurrency limiter — max 3 parallel Gemini calls
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
        const response = await ai.models.generateContent({
            model: 'gemini-3.1-flash-image-preview',
            contents: [{ parts }],
            config: { responseModalities: ['TEXT', 'IMAGE'] },
        });

        const resParts  = response.candidates?.[0]?.content?.parts || [];
        const imagePart = resParts.find(p => p.inlineData?.data && !p.thought);
        if (!imagePart) {
            const text = resParts.find(p => p.text)?.text || 'none';
            console.error('[Gemini] No image. Response text:', text.slice(0, 300));
            throw new Error('Gemini returned no image \u2014 ' + text.slice(0, 120));
        }

        // Validate the returned image
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
app.listen(PORT, () => console.log(`\n\ud83d\ude80  Image Pipeline \u2192 http://localhost:${PORT}\n`));
