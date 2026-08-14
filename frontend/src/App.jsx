import { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";

import ekaAPI from "@/api/ekaClient";
import Sidebar from "@/components/Sidebar";
import Chat from "@/pages/Chat";
import Goals from "@/pages/Goals";
import Memory from "@/pages/Memory";
import Reflections from "@/pages/Reflections";
import Settings from "@/pages/Settings";

/**
 * Shell: persistent sidebar, routed content.
 *
 * `mode` lives here rather than in Chat because it is app-level — the sidebar
 * sets it, the chat header reflects it, and Settings edits it. It persists so a
 * reload does not silently drop you back into founder.
 */
export default function App() {
  const [mode, setMode] = useState(
    () => localStorage.getItem("eka_mode") || "founder"
  );
  const [health, setHealth] = useState("checking");

  useEffect(() => {
    localStorage.setItem("eka_mode", mode);
  }, [mode]);

  useEffect(() => {
    let alive = true;
    ekaAPI
      .checkHealth()
      .then(() => alive && setHealth("up"))
      .catch(() => alive && setHealth("down"));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="flex h-full">
      <Sidebar mode={mode} onMode={setMode} health={health} />
      <div className="min-w-0 flex-1">
        <Routes>
          <Route path="/" element={<Chat mode={mode} />} />
          <Route path="/memory" element={<Memory />} />
          <Route path="/goals" element={<Goals />} />
          <Route path="/reflections" element={<Reflections />} />
          <Route
            path="/settings"
            element={<Settings mode={mode} onMode={setMode} />}
          />
        </Routes>
      </div>
    </div>
  );
}
