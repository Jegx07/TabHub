const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const root = path.join(__dirname, "..");
const catalog = JSON.parse(fs.readFileSync(path.join(root, "medicine_metadata.json"), "utf8"));

async function signature(buffer) {
  const { data } = await sharp(buffer)
    .resize(64, 64, { fit: "fill" })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return data;
}

function score(left, right) {
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference += Math.abs(left[index] - right[index]);
  }
  return 100 - (difference / (left.length * 255)) * 100;
}

function reportFor(record, matchScore) {
  const composition = (record.composition || [])
    .map((item) => `- ${item.ingredient || "Unknown"}: ${item.strength || "Unverified"}`)
    .join("\n") || "- Composition unavailable.";
  return `### SECTION A: IDENTIFICATION

- **Status:** \`IDENTIFIED\` (${matchScore.toFixed(1)}/100 Reference Match)
- **Brand Name:** **${record.brand_name || "Unknown"}**
- **Generic Name:** ${record.generic_name || "Unknown"}
- **Dosage Form:** ${record.dosage_form || "Unverified"}
- **Manufacturer:** ${record.manufacturer || "Unverified"}

### SECTION B: COMPOSITION

${composition}

### SECTION C: USES & APPLICATIONS

**Uses:** ${record.uses || "Verify approved indications with a clinician."}

**Mechanism:** ${record.mechanism_of_action || "Verify with an official monograph."}

### SECTION D: SAFETY

**Side effects:** ${record.side_effects || "See official product information."}

**Serious warnings:** ${record.serious_adverse_effects || "Seek medical advice for severe reactions."}

**Precautions:** ${record.precautions || "Use only as directed by a healthcare professional."}

> Informational use only. Confirm the medicine with a licensed pharmacist or physician.`;
}

module.exports = async function handler(req, res) {
  if (req.method === "GET") {
    return res.status(200).json({ status: "ok", service: "MediScan AI" });
  }
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed." });
  }
  try {
    const encoded = req.body && req.body.image;
    if (typeof encoded !== "string" || !encoded) {
      return res.status(400).json({ error: "A base64-encoded image is required." });
    }
    const image = Buffer.from(encoded.includes(",") ? encoded.split(",", 2)[1] : encoded, "base64");
    if (!image.length || image.length > 12 * 1024 * 1024) {
      return res.status(413).json({ error: "Image must be between 1 byte and 12 MB." });
    }
    const target = await signature(image);
    const matches = [];
    for (const [medicineId, record] of Object.entries(catalog)) {
      const references = record.reference_images || [record.reference_image];
      let best = 0;
      for (const reference of references.filter(Boolean)) {
        const referencePath = path.join(root, reference);
        if (fs.existsSync(referencePath)) {
          best = Math.max(best, score(target, await signature(referencePath)));
        }
      }
      if (best >= 85) matches.push({ medicineId, record, score: best });
    }
    matches.sort((a, b) => b.score - a.score);
    if (!matches.length || (matches[1] && matches[0].score - matches[1].score < 5)) {
      return res.status(200).json({
        report: "### SECTION A: IDENTIFICATION\n\n- **Status:** `UNIDENTIFIED`\n- **Message:** Medicine could not be identified reliably. Please upload a clearer packaging or label image.",
      });
    }
    return res.status(200).json({ report: reportFor(matches[0].record, matches[0].score) });
  } catch (error) {
    return res.status(400).json({ error: `Invalid image request: ${error.message}` });
  }
};
