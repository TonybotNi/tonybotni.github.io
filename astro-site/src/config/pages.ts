import type { PagesConfig } from "../types";

export const PAGES: PagesConfig = {
    home: {
        title: "About Me",
        subtitle: "",
        isActive: true,
    },
    blog: {
        title: "Blog",
        subtitle: "",
        isActive: true,
    },
    publications: {
        title: "Publications",
        subtitle: "Research papers and scientific articles.",
        isActive: true,
    },
    talks: {
        title: "Talks & Presentations",
        subtitle: "",
        isActive: false,
    },
    projects: {
        title: "Projects",
        subtitle: "Competitions and open source contributions.",
        isActive: true,
    },
    teaching: {
        title: "Teaching",
        subtitle: "",
        isActive: false,
    },
    tags: {
        title: "Tags",
        subtitle: "",
        isActive: false,
    },
    cv: {
        title: "Curriculum Vitae",
        subtitle: "",
        isActive: false,
    },
};
