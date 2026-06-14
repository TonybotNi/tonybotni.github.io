import { SITE } from "../config";

/**
 * Resolve an image path to its final URL.
 * - Absolute URLs (http/https) are returned unchanged.
 * - When SITE.assetsBase is set, local paths like "/images/x.png" are prefixed
 *   with the CDN/COS base (e.g. COS "Homepage" folder).
 * - When SITE.assetsBase is empty, the original local path is kept so the site
 *   still works in offline/local development.
 */
export function toAssetUrl(path?: string): string | undefined {
    if (!path) return path;
    if (/^https?:\/\//i.test(path)) return path;
    const base = SITE.assetsBase?.replace(/\/+$/, "");
    if (!base) return path;
    return `${base}${path.startsWith("/") ? "" : "/"}${path}`;
}
