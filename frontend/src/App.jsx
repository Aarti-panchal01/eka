import { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";

import ekaAPI from "@/api/ekaClient";
import Sidebar from "@/components/Sidebar";
import { getLanguage, setLanguage as persistLanguage } from "@/lib/ui";
import Chat from "@/pages/Chat";
import Goals from "@/pages/Goals";
import Memory from "@/pages/Memory";
import Reflections from "@/pages/Reflections";
import Settings from "@/pages/Settings";

/**
 * Shell: persistent sidebar, routed content.
 *
 * `mode` and `language` live here because they are app-level — the sidebar sets
 * mode, the chat header shows both, and Settings edits both. Both persist, so a
 * reload does not silently drop you back to founder-in-English.
 */
export default function App() {
  const [mode, setMode] = useState(
    () => localStorage.getItem("eka_mode") || "founder"
  );
  const [language, setLanguage] = useState(getLanguage);
  const [health, setHealth] = useState("checking");
  // Bumped by "New chat". A token rather than a callback keeps Chat's reset
  // logic inside Chat, where the session state lives.
  const [newChatToken, setNewChatToken] = useState(0);

  useEffect(() => {
    localStorage.setItem("eka_mode", mode);
  }, [mode]);

  useEffect(() => {
    persistLanguage(language);
  }, [language]);

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
      <Sidebar
        mode={mode}
        onMode={setMode}
        health={health}
        onNewChat={() => setNewChatToken((n) => n + 1)}
      />
      <div className="min-w-0 flex-1">
        <Routes>
          <Route
            path="/"
            element={
              <Chat
                mode={mode}
                language={language}
                health={health}
                newChatToken={newChatToken}
              />
            }
          />
          <Route path="/memory" element={<Memory />} />
          <Route path="/goals" element={<Goals />} />
          <Route path="/reflections" element={<Reflections />} />
          <Route
            path="/settings"
            element={
              <Settings
                mode={mode}
                onMode={setMode}
                language={language}
                onLanguage={setLanguage}
              />
            }
          />
        </Routes>
      </div>
    </div>
  );
}
