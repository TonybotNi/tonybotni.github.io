#!/usr/bin/env node
/**
 * Upload homepage images to Tencent COS under the "Homepage/" prefix.
 *
 * Mapping (keeps the same layout the site expects):
 *   public/avatar.jpg        -> Homepage/avatar.jpg
 *   public/images/<file>     -> Homepage/images/<file>
 *
 * Credentials are read from environment variables (never hardcoded):
 *   COS_SECRET_ID, COS_SECRET_KEY   (required)
 *   COS_BUCKET   (default: pic-1313147768)
 *   COS_REGION   (default: ap-chengdu)
 *   COS_PREFIX   (default: Homepage)
 *
 * Usage:
 *   COS_SECRET_ID=xxx COS_SECRET_KEY=yyy npm run upload:images
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import COS from "cos-nodejs-sdk-v5";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.resolve(__dirname, "..", "public");

const SecretId = process.env.COS_SECRET_ID;
const SecretKey = process.env.COS_SECRET_KEY;
const Bucket = process.env.COS_BUCKET || "pic-1313147768";
const Region = process.env.COS_REGION || "ap-chengdu";
const Prefix = (process.env.COS_PREFIX || "Homepage").replace(/\/+$/, "");

if (!SecretId || !SecretKey) {
    console.error(
        "\n❌ Missing credentials. Set COS_SECRET_ID and COS_SECRET_KEY, e.g.\n" +
            "   COS_SECRET_ID=xxx COS_SECRET_KEY=yyy npm run upload:images\n"
    );
    process.exit(1);
}

const MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
};

const IMAGE_EXTS = new Set(Object.keys(MIME));

/** Collect { local, key } pairs to upload. */
function collectFiles() {
    const jobs = [];

    const avatar = path.join(PUBLIC_DIR, "avatar.jpg");
    if (fs.existsSync(avatar)) {
        jobs.push({ local: avatar, key: `${Prefix}/avatar.jpg` });
    }

    const imagesDir = path.join(PUBLIC_DIR, "images");
    if (fs.existsSync(imagesDir)) {
        for (const name of fs.readdirSync(imagesDir)) {
            const local = path.join(imagesDir, name);
            if (!fs.statSync(local).isFile()) continue;
            if (!IMAGE_EXTS.has(path.extname(name).toLowerCase())) continue;
            jobs.push({ local, key: `${Prefix}/images/${name}` });
        }
    }

    return jobs;
}

const cos = new COS({ SecretId, SecretKey });

function putObject({ local, key }) {
    const ext = path.extname(local).toLowerCase();
    const size = fs.statSync(local).size;
    return new Promise((resolve, reject) => {
        cos.putObject(
            {
                Bucket,
                Region,
                Key: key,
                Body: fs.createReadStream(local),
                ContentLength: size,
                ContentType: MIME[ext] || "application/octet-stream",
                ACL: "public-read",
            },
            (err, data) => (err ? reject(err) : resolve(data))
        );
    });
}

async function main() {
    const jobs = collectFiles();
    if (jobs.length === 0) {
        console.error("No images found under public/. Nothing to upload.");
        process.exit(1);
    }

    console.log(
        `Uploading ${jobs.length} file(s) to cos://${Bucket} (${Region}) under "${Prefix}/" ...\n`
    );

    let ok = 0;
    for (const job of jobs) {
        try {
            await putObject(job);
            ok++;
            console.log(`  ✓ ${path.relative(PUBLIC_DIR, job.local)}  ->  ${job.key}`);
        } catch (err) {
            console.error(`  ✗ ${job.key}  (${err?.message || err})`);
        }
    }

    const base = `https://${Bucket}.cos.${Region}.myqcloud.com/${Prefix}`;
    console.log(`\nDone. ${ok}/${jobs.length} uploaded.`);
    console.log(`Verify: ${base}/avatar.jpg`);
    if (ok !== jobs.length) process.exit(1);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
