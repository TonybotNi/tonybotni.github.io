export function isChinesePostId(id: string): boolean {
    return id.endsWith("-zh");
}

export function getAlternatePostId(id: string): string {
    return isChinesePostId(id) ? id.slice(0, -3) : `${id}-zh`;
}

export function filterEnglishPosts<T extends { id: string }>(posts: T[]): T[] {
    return posts.filter((p) => !isChinesePostId(p.id));
}

export function sortPostsByDate<T extends { data: { date?: string } }>(posts: T[]): T[] {
    return [...posts].sort(
        (a, b) => new Date(b.data.date || 0).getTime() - new Date(a.data.date || 0).getTime(),
    );
}
