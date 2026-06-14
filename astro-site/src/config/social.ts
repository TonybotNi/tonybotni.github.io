import type { SocialLink } from "../types";

export const SOCIALS: SocialLink[] = [
    {
        name: "Github",
        href: "https://github.com/TonybotNi",
        linkTitle: `Jingchen Ni on Github`,
        isActive: true,
    },
    {
        name: "Mail",
        href: "mailto:njc24@mails.tsinghua.edu.cn",
        linkTitle: `Send an email to Jingchen Ni`,
        isActive: true,
    },
    {
        name: "Google Scholar",
        href: "https://scholar.google.com/citations?user=KJSB8EkAAAAJ&hl=en",
        linkTitle: `Jingchen Ni on Google Scholar`,
        isActive: true,
    },
];

export const SOCIAL_ICONS: Record<string, string> = {
    Github: "Github",
    Mail: "Mail",
    Linkedin: "LinkedIn",
    "Google Scholar": "GoogleScholar",
    ORCID: "ORCID",
    RSS: "RSS",
};