/**
 * Site footer. Deliberately quiet — one line, small, low contrast, so it never
 * competes with the conversation above it. Identity left, attribution and
 * links grouped right so the eye lands on one cluster rather than three.
 */
export default function Footer() {
  return (
    <footer className="shrink-0 border-t border-edge bg-sidebar px-6 py-3">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-[#888888]">
        <span className="flex items-center gap-1.5">
          <span className="text-gold">◆</span> Eka — Your lifelong AI companion
        </span>

        <span className="ml-auto flex items-center gap-3">
          <span>Built by Aarti Panchal</span>
          <span className="text-edge">|</span>

          {/* rel=noreferrer alongside _blank: without it the opened tab gets a
              handle back to this one via window.opener. */}
          <a
            href="https://linkedin.com/in/aarti-panchal"
            target="_blank"
            rel="noopener noreferrer"
            title="LinkedIn"
            className="transition-all duration-200 hover:text-gold"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <path d="M4.98 3.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5zM3 9h4v12H3zM9 9h3.8v1.7h.05c.53-1 1.83-2.05 3.77-2.05 4.03 0 4.78 2.65 4.78 6.1V21h-4v-5.5c0-1.3-.02-3-1.83-3-1.83 0-2.11 1.43-2.11 2.9V21H9z" />
            </svg>
            <span className="sr-only">LinkedIn</span>
          </a>

          <a
            href="https://github.com/Aarti-panchal01"
            target="_blank"
            rel="noopener noreferrer"
            title="GitHub"
            className="transition-all duration-200 hover:text-gold"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <path d="M12 .5C5.73.5.9 5.33.9 11.6c0 4.9 3.18 9.06 7.6 10.53.56.1.76-.24.76-.53v-2.06c-3.1.67-3.75-1.3-3.75-1.3-.5-1.3-1.24-1.64-1.24-1.64-1.02-.7.08-.68.08-.68 1.13.08 1.72 1.16 1.72 1.16 1 1.72 2.63 1.22 3.27.93.1-.73.4-1.22.72-1.5-2.48-.28-5.08-1.24-5.08-5.5 0-1.22.43-2.21 1.15-2.99-.12-.28-.5-1.42.1-2.96 0 0 .94-.3 3.08 1.14a10.6 10.6 0 0 1 5.6 0c2.14-1.44 3.07-1.14 3.07-1.14.6 1.54.23 2.68.11 2.96.72.78 1.15 1.77 1.15 2.99 0 4.27-2.6 5.21-5.09 5.49.4.35.77 1.03.77 2.08v3.09c0 .3.2.64.77.53a11.1 11.1 0 0 0 7.59-10.53C23.1 5.33 18.27.5 12 .5z" />
            </svg>
            <span className="sr-only">GitHub</span>
          </a>
        </span>
      </div>
    </footer>
  );
}
