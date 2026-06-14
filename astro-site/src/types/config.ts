export interface SiteConfig {
    website: string;
    author: string;
    desc: string;
    title: string;
    ogImage: string;
    postPerPage: number;
    favicon: string;
    lang: string;
    /** Base URL for hosted assets (e.g. Tencent COS "Homepage" folder). Empty = use local /public paths. */
    assetsBase: string;
}

export interface ThemeConfig {
    lightAndDark: boolean;
    themeLight: string;
    themeDark: string;
}

export interface SettingsConfig {
    showTagsInNavbar: boolean;
    showRSSInFooter: boolean;
    addDevToolsInProduction: boolean;
}

export interface UmamiAnalyticsConfig {
    websiteId: string;
    src: string;
}

export interface AnalyticsConfig {
    ga4Id?: string;
    umami?: UmamiAnalyticsConfig;
}

export interface NavLink {
    href: string;
    label: string;
    isActive: boolean;
}

export interface SocialLink {
    name: string;
    href: string;
    linkTitle: string;
    isActive: boolean;
}

export interface PageConfig {
    title: string;
    subtitle: string;
    isActive: boolean;
}

export type PagesConfig = Record<string, PageConfig>;