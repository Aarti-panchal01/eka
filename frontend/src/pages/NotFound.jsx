import { Link, useLocation } from "react-router-dom";

import { NAV, PageHeader } from "@/lib/ui";

/**
 * Catch-all for unknown paths.
 *
 * Without this, an unmatched route rendered nothing into the shell — sidebar
 * and footer painted, the middle stayed empty — which reads as a broken app
 * rather than a wrong URL. Anything the router does not know now says so.
 *
 * The destinations come from NAV rather than a second hand-written list, so a
 * page added to the sidebar shows up here without anyone remembering to.
 */
export default function NotFound() {
  const { pathname } = useLocation();

  return (
    <div className="scroll-thin h-full overflow-y-auto px-8 py-7">
      <div className="mx-auto max-w-4xl">
        <PageHeader
          title="Nothing here"
          subtitle={`No page at ${pathname}`}
        />

        <div className="card p-6">
          <p className="text-sm text-neutral-400">
            The link is wrong or the page moved. Everything Eka can do:
          </p>

          <div className="mt-5 grid gap-2 sm:grid-cols-2">
            {NAV.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className="flex items-center gap-3 rounded-lg border border-edge
                           px-4 py-3 transition-all duration-200
                           hover:border-neutral-600"
              >
                <span className="text-lg">{item.icon}</span>
                <span className="min-w-0">
                  <span className="block text-sm font-semibold text-neutral-100">
                    {item.label}
                  </span>
                  <span className="block text-xs text-neutral-500">
                    {item.hint}
                  </span>
                </span>
              </Link>
            ))}
          </div>

          <Link to="/" className="btn-gold mt-6 inline-block">
            Back to chat
          </Link>
        </div>
      </div>
    </div>
  );
}
