import React, { useEffect, useState } from "react";
import { useNativeMessaging } from "../hooks/useNativeMessaging";

function Popup() {
  const { isConnected, connect, analyzeDom, sendMessage } = useNativeMessaging();
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { connect(); }, []);

  const sendHeartBeatEvent = async () => {
    const result = await sendMessage({ type: "heartbeat" });
    console.log("Heartbeat result → ", result);
  };

  const verifyUrl = async () => {
    setLoading(true);
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.url) return;
      const result = await analyzeDom(tab.id);
      console.log("DOM verification result →", result);
      if (result?.result) {
        setAnalysisResult(result.result);
      } else if (result?.success === false) {
        const urlResult = await sendMessage({ type: "url_analysis", url: tab.url });
        setAnalysisResult(urlResult);
      }
    } catch (error) {
      console.error("Failed to verify URL:", error);
    } finally {
      setLoading(false);
    }
  };

  const isSafe = analysisResult?.page_classification !== "bad";
  const confidence = analysisResult ? (analysisResult.page_confidence * 100).toFixed(1) : null;

  return (
    <div className="min-h-[460px] bg-gradient-to-b from-slate-50 to-slate-100 flex flex-col">

      <header className="border-b border-slate-200 bg-white px-5 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="size-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-sm">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-tight text-slate-900">SafetyPin</h1>
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

      <div className="flex-1 overflow-y-auto px-5 py-4">
        {analysisResult && (
          <div className="rounded-2xl bg-white border border-slate-200/80 shadow-sm p-4 space-y-4 py-2">
            <div className={`rounded-xl px-3.5 py-3 border ${
              isSafe
                ? "bg-emerald-50 border-emerald-200/70"
                : "bg-rose-50 border-rose-200/70"
            }`}>
              <div className="flex items-start gap-2.5">
                <div className={`mt-0.5 size-5 shrink-0 ${isSafe ? "text-emerald-600" : "text-rose-600"}`}>
                  {isSafe ? (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-semibold tracking-tight ${isSafe ? "text-emerald-900" : "text-rose-900"}`}>
                    {isSafe ? "Page looks safe" : "Unsafe content detected"}
                  </p>
                  <p className={`mt-1 text-xs font-medium ${isSafe ? "text-emerald-700" : "text-rose-700"}`}>
                    {analysisResult.page_classification?.toUpperCase()} &middot; {confidence}% confidence
                  </p>
                </div>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-slate-600">Confidence</span>
                <span className="text-xs font-semibold text-slate-700">{confidence}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${isSafe ? "bg-emerald-500" : "bg-rose-500"}`}
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-2.5">
              {analysisResult.image_analysis && (
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-3 flex items-center gap-3">
                  <div className="size-8 rounded-lg bg-violet-100 text-violet-700 flex items-center justify-center shrink-0">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-slate-900">Image scan</p>
                    <p className="text-[11px] text-slate-500">{analysisResult.image_analysis.images_analyzed} analyzed</p>
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] shrink-0">
                    <span className="inline-flex rounded-md bg-rose-100 text-rose-700 px-1.5 py-0.5 font-semibold">{analysisResult.image_analysis.bad_images}</span>
                    <span className="inline-flex rounded-md bg-emerald-100 text-emerald-700 px-1.5 py-0.5 font-semibold">{analysisResult.image_analysis.good_images}</span>
                  </div>
                </div>
              )}

              {analysisResult.text_analysis && (
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-3 flex items-start gap-3">
                  <div className="size-8 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center shrink-0 mt-0.5">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs font-semibold text-slate-900">Text analysis</p>
                      <span className={`inline-flex rounded-md px-1.5 py-0.5 text-[11px] font-semibold ${
                        analysisResult.text_analysis.classification === "bad"
                          ? "bg-rose-100 text-rose-700"
                          : "bg-emerald-100 text-emerald-700"
                      }`}>
                        {analysisResult.text_analysis.classification}
                      </span>
                    </div>
                    {analysisResult.text_analysis.reason && (
                      <p className="mt-1 text-[11px] leading-4 text-slate-500 line-clamp-2">{analysisResult.text_analysis.reason}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {!analysisResult && !loading && (
          <div className="h-full min-h-[220px] flex flex-col items-center justify-center text-center">
            <div className="size-12 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center mb-3">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </div>
            <p className="text-sm font-semibold text-slate-900">Ready to scan this page</p>
            <p className="mt-1 text-xs text-slate-500">Run verification to see a full safety report.</p>
          </div>
        )}

        {loading && !analysisResult && (
          <div className="h-full min-h-[220px] flex flex-col items-center justify-center text-center">
            <div className="size-10 mb-3">
              <svg className="animate-spin text-indigo-600" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
            </div>
            <p className="text-sm font-semibold text-slate-900">Analyzing page…</p>
            <p className="mt-1 text-xs text-slate-500">Checking text and media content</p>
          </div>
        )}
      </div>

      <div className="border-t border-slate-200/80 bg-white/90 backdrop-blur px-5 pt-5 pb-6">
        <div className="flex justify-center">
          <button
            onClick={verifyUrl}
            disabled={loading}
            className="inline-flex min-w-[230px] items-center justify-center rounded-2xl bg-indigo-600 px-10 py-4 text-sm font-semibold tracking-tight text-white shadow-[0_10px_20px_rgba(79,70,229,0.28)] hover:bg-indigo-500 transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:bg-indigo-400 disabled:cursor-not-allowed"
          >
            {loading ? "Scanning…" : "Verify Current Page"}
          </button>
        </div>

        <div className="mt-5 flex justify-center">
          <button
            onClick={sendHeartBeatEvent}
            disabled={loading}
            className="inline-flex items-center rounded-lg px-3 py-2 text-xs font-medium text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send Heartbeat
          </button>
        </div>
      </div>
    </div>
  );
}

export default Popup;
