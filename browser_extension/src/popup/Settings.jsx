import React, { useEffect, useState } from "react";
import { Shield } from "lucide-react";
import { useNativeMessaging } from "../hooks/useNativeMessaging";

const HISTORY_FILTERS = [
  { value: "today", label: "Today" },
  { value: "yesterday", label: "Yesterday" },
  { value: "week", label: "This week" },
  { value: "month", label: "This month" },
  { value: "all", label: "All" },
];

function Settings() {
  const { isConnected, connect, sendMessage } = useNativeMessaging();
  const [notificationEmail, setNotificationEmail] = useState("");
  const [notificationPhone, setNotificationPhone] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const [nativeLoadMessage, setNativeLoadMessage] = useState("");
  const [historyItems, setHistoryItems] = useState([]);
  const [historyFilter, setHistoryFilter] = useState("today");
  const [historySearch, setHistorySearch] = useState("");

  useEffect(() => {
    const loadFromLocalStorage = () => {
      if (typeof chrome !== "undefined" && chrome.storage?.local) {
        chrome.storage.local.get(["notificationEmail", "notificationPhone"], (result) => {
          if (typeof result.notificationEmail === "string") {
            setNotificationEmail(result.notificationEmail);
          }

          if (typeof result.notificationPhone === "string") {
            setNotificationPhone(result.notificationPhone);
          }
        });
      }
    };

    const loadUserInfo = async () => {
      try {
        const nativeResult = await sendMessage({ type: "get_user_info" });

        if (nativeResult?.error) {
          setNativeLoadMessage("Loaded from local storage (native DB unavailable).");
          loadFromLocalStorage();
          return;
        }

        const email = typeof nativeResult?.email === "string" ? nativeResult.email : "";
        const phone = typeof nativeResult?.phone === "string" ? nativeResult.phone : "";

        if (!email && !phone) {
          setNativeLoadMessage("No native profile found. Showing local values.");
          loadFromLocalStorage();
          return;
        }

        setNotificationEmail(email);
        setNotificationPhone(phone);

        if (typeof chrome !== "undefined" && chrome.storage?.local) {
          await chrome.storage.local.set({
            notificationEmail: email,
            notificationPhone: phone,
          });
        }

        setNativeLoadMessage("Loaded Profile Information");
      } catch {
        setNativeLoadMessage("Loaded from local storage (native DB unavailable).");
        loadFromLocalStorage();
      }
    };

    const loadHistory = async () => {
      try {
        const historyResult = await sendMessage({ type: "get_low_reputation_history", limit: 1000 });
        if (historyResult?.error) {
          setHistoryItems([]);
          return;
        }

        const items = Array.isArray(historyResult?.items) ? historyResult.items : [];
        setHistoryItems(items);
      } catch {
        setHistoryItems([]);
      }
    };

    const initializeSettings = async () => {
      const connectionResult = await connect();
      if (!connectionResult?.success) {
        setNativeLoadMessage("Loaded from local storage (native DB unavailable).");
        loadFromLocalStorage();
        setHistoryItems([]);
        return;
      }

      await loadUserInfo();
      await loadHistory();
    };

    initializeSettings();
  }, [connect, sendMessage]);

  const formatTimestamp = (unixSeconds) => {
    const value = Number(unixSeconds || 0);
    if (!value) return "Unknown time";
    return new Date(value * 1000).toLocaleString();
  };

  const getFilteredHistoryItems = () => {
    if (!Array.isArray(historyItems)) {
      return [];
    }

    const normalize = (value) => String(value || "").toLowerCase().trim();
    const searchQuery = normalize(historySearch);
    const numericSearchMatch = searchQuery.match(/^\d+(\.\d+)?%?$/);
    const minProtectionScore = numericSearchMatch
      ? Math.min(Math.max(Number.parseFloat(searchQuery.replace("%", "")), 0), 100)
      : null;
    const isFuzzyMatch = (target, query) => {
      if (!query) return true;
      const normalizedTarget = normalize(target);
      if (!normalizedTarget) return false;

      let queryIndex = 0;
      for (let i = 0; i < normalizedTarget.length && queryIndex < query.length; i += 1) {
        if (normalizedTarget[i] === query[queryIndex]) {
          queryIndex += 1;
        }
      }

      return queryIndex === query.length;
    };

    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startOfTomorrow = new Date(startOfToday);
    startOfTomorrow.setDate(startOfTomorrow.getDate() + 1);

    const startOfYesterday = new Date(startOfToday);
    startOfYesterday.setDate(startOfYesterday.getDate() - 1);

    const dayOfWeek = startOfToday.getDay();
    const daysFromMonday = (dayOfWeek + 6) % 7;
    const startOfWeek = new Date(startOfToday);
    startOfWeek.setDate(startOfWeek.getDate() - daysFromMonday);

    const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);

    return historyItems.filter((item) => {
      const ts = Number(item?.created_at || 0);
      const safetyPct = Number(item?.safety_score ?? 0) * 100;

      if (minProtectionScore !== null) {
        if (Number.isNaN(safetyPct) || safetyPct < minProtectionScore) {
          return false;
        }
      }

      const matchesWebsite =
        minProtectionScore !== null
          ? true
          : isFuzzyMatch(item?.domain, searchQuery) || isFuzzyMatch(item?.visited_url, searchQuery);
      if (!matchesWebsite) return false;

      if (!ts) {
        return historyFilter === "all";
      }

      const itemDate = new Date(ts * 1000);

      switch (historyFilter) {
        case "today":
          return itemDate >= startOfToday && itemDate < startOfTomorrow;
        case "yesterday":
          return itemDate >= startOfYesterday && itemDate < startOfToday;
        case "week":
          return itemDate >= startOfWeek;
        case "month":
          return itemDate >= startOfMonth;
        case "all":
        default:
          return true;
      }
    });
  };

  const filteredHistoryItems = getFilteredHistoryItems();

  const saveNotificationContacts = async () => {
    if (typeof chrome === "undefined" || !chrome.storage?.local) {
      setSaveMessage("Unable to save on this page.");
      return;
    }

    await chrome.storage.local.set({
      notificationEmail: notificationEmail.trim(),
      notificationPhone: notificationPhone.trim(),
    });

    const connectionResult = await connect();
    if (!connectionResult?.success) {
      setSaveMessage("Saved locally, but native save failed.");
      return;
    }

    try {
      const nativeResult = await sendMessage({
        type: "save_user_info",
        email: notificationEmail.trim(),
        phone: notificationPhone.trim(),
      });

      if (nativeResult?.error) {
        setSaveMessage("Saved locally, but native save failed.");
        return;
      }
    } catch {
      setSaveMessage("Saved locally, but native save failed.");
      return;
    }

    setSaveMessage("Notification contacts saved.");
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-50 to-slate-100 flex flex-col">
      <header className="border-b border-slate-200 bg-white px-5 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="size-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-sm">
              <Shield className="size-4 text-white" strokeWidth={2.5} />
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-tight text-slate-900">SafeNest</h1>
              <p className="text-[11px] text-slate-500">Child safety monitor</p>
            </div>
          </div>

          <div className={`flex items-center gap-1.5 text-xs font-medium ${
            isConnected ? "text-emerald-700" : "text-rose-700"
          }`}>
            <span className={`size-1.5 rounded-full ${isConnected ? "bg-emerald-500" : "bg-rose-500"}`} />
            {isConnected ? "Connected" : "Offline"}
          </div>
        </div>
      </header>

      <div className="flex-1 w-full overflow-y-auto">
        <div className="mx-auto w-full max-w-5xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="rounded-xl border border-slate-200 bg-white p-5 sm:p-6">
            <p className="text-base font-semibold text-slate-900">Settings</p>
            <p className="mt-1 text-sm text-slate-600">Configure SafeNest behavior for your family’s browsing experience.</p>
            {nativeLoadMessage && <p className="mt-2 text-xs text-indigo-700">{nativeLoadMessage}</p>}
          </div>

          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-1">

            <div className="rounded-xl border border-slate-200 bg-white p-5 sm:p-6">
              <p className="text-sm font-semibold text-slate-900">Notifications</p>
              <p className="mt-1 text-sm text-slate-600">Add where SafeNest should send alerts and reports.</p>

              <div className="mt-4 space-y-3">
                <div>
                  <label htmlFor="notification-email" className="block text-xs font-medium text-slate-700">Email</label>
                  <input
                    id="notification-email"
                    type="email"
                    value={notificationEmail}
                    onChange={(event) => setNotificationEmail(event.target.value)}
                    placeholder="parent@example.com"
                    className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                  />
                </div>

                <div>
                  <label htmlFor="notification-phone" className="block text-xs font-medium text-slate-700">Phone number</label>
                  <input
                    id="notification-phone"
                    type="tel"
                    value={notificationPhone}
                    onChange={(event) => setNotificationPhone(event.target.value)}
                    placeholder="+1 555 123 4567"
                    className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                  />
                </div>

                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs text-slate-500">Used for safety notifications only.</p>
                  <button
                    type="button"
                    onClick={saveNotificationContacts}
                    className="inline-flex items-center rounded-lg bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500"
                  >
                    Save
                  </button>
                </div>

                {saveMessage && <p className="text-xs text-emerald-700">{saveMessage}</p>}
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-5 sm:p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold text-slate-900">History</p>
                  <p className="mt-1 text-sm text-slate-600">Sites visited with safety score below 70%.</p>
                </div>

                <div className="w-full max-w-[520px]">
                  <label htmlFor="history-filter" className="block text-xs font-medium text-slate-700 text-right">Filter</label>
                  <div className="mt-1 flex items-center gap-2">
                    <input
                      id="history-search"
                      type="text"
                      value={historySearch}
                      onChange={(event) => setHistorySearch(event.target.value)}
                      placeholder="Search websites or min score (e.g. 70 or 70%)"
                      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                    />
                    <select
                      id="history-filter"
                      value={historyFilter}
                      onChange={(event) => setHistoryFilter(event.target.value)}
                      className="w-full max-w-[220px] rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                    >
                      {HISTORY_FILTERS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <p className="mt-1 text-xs text-slate-500 text-right">Tip: enter a number to show sites with protection score at or above that value.</p>
                </div>
              </div>

              <div className="mt-4 max-h-[440px] overflow-y-auto pr-1 space-y-3">
                {filteredHistoryItems.length === 0 && (
                  <p className="text-sm text-slate-500">No low-reputation sites recorded yet.</p>
                )}

                {filteredHistoryItems.map((item, index) => {
                  const safetyPct = Number(item?.safety_score ?? 0) * 100;
                  const isBlocked = item?.blocked_page_shown === true || item?.action_taken === "page_blocked";
                  return (
                    <div key={`${item?.visited_url || "visit"}-${item?.created_at || index}-${index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <p className="text-sm font-semibold text-slate-900">{item?.domain || "unknown domain"}</p>
                      <p className="mt-1 text-xs text-slate-600 break-all">{item?.visited_url || ""}</p>
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                        <span className="rounded bg-rose-100 px-2 py-0.5 font-medium text-rose-700">Safety: {safetyPct.toFixed(1)}%</span>
                        <span className="rounded bg-slate-200 px-2 py-0.5 font-medium text-slate-700">{item?.analysis_type || "analysis"}</span>
                        <span className="rounded bg-slate-200 px-2 py-0.5 font-medium text-slate-700">{item?.page_classification || "unknown"}</span>
                        {isBlocked && <span className="rounded bg-red-600 px-2 py-0.5 font-semibold text-white">Blocked</span>}
                      </div>
                      <p className="mt-2 text-xs text-slate-500">Visited: {formatTimestamp(item?.created_at)}</p>
                      {item?.reasons && <p className="mt-1 text-xs text-slate-500">Reasons: {item.reasons}</p>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
