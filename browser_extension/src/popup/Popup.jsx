import React, { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, FileText, Globe, ImageIcon, Loader2, Shield, HeartPlus, Settings } from "lucide-react";
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
  const domainRisk = typeof analysisResult?.domain_analysis?.risk_score === "number"
    ? analysisResult.domain_analysis.risk_score
    : null;
  const domainSafety = typeof analysisResult?.domain_analysis?.risk_score === "number"
    ? ((1 - analysisResult.domain_analysis.risk_score) * 100).toFixed(1)
    : null;
  const badImageItems = analysisResult?.image_analysis?.bad_image_elements || [];
  const badTextItems = analysisResult?.text_analysis?.bad_text_elements || [];
  const domainTone = domainRisk === null
    ? {
        iconBg: "bg-slate-100",
        iconText: "text-slate-700",
        badgeBg: "bg-slate-100",
        badgeText: "text-slate-700",
      }
    : domainRisk >= 0.9
      ? {
          iconBg: "bg-rose-100",
          iconText: "text-rose-700",
          badgeBg: "bg-rose-100",
          badgeText: "text-rose-700",
        }
      : domainRisk >= 0.5
        ? {
            iconBg: "bg-amber-100",
            iconText: "text-amber-700",
            badgeBg: "bg-amber-100",
            badgeText: "text-amber-700",
          }
        : {
            iconBg: "bg-emerald-100",
            iconText: "text-emerald-700",
            badgeBg: "bg-emerald-100",
            badgeText: "text-emerald-700",
          };

  const openDetailsPage = async (section) => {
    if (!analysisResult) return;

    const items = section === "images" ? badImageItems : badTextItems;
    if (!items.length) return;

    if (typeof chrome === "undefined" || !chrome.storage?.local || !chrome.tabs?.create || !chrome.runtime?.getURL) {
      return;
    }

    const reportId = `finding_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const reportPayload = {
      section,
      createdAt: Date.now(),
      requestedUrl: analysisResult.requested_url || "",
      domain: analysisResult?.domain_analysis?.domain || "",
      category: analysisResult?.domain_analysis?.category || "unknown",
      items,
    };

    await chrome.storage.local.set({ [reportId]: reportPayload });
    const detailsUrl = chrome.runtime.getURL(`blocked.html?view=details&section=${encodeURIComponent(section)}&id=${encodeURIComponent(reportId)}`);
    await chrome.tabs.create({ url: detailsUrl });
  };

  const openSettingsPopup = async () => {
    if (typeof chrome === "undefined" || !chrome.runtime?.getURL || !chrome.tabs?.create) {
      return;
    }

    const settingsUrl = chrome.runtime.getURL("settings.html");
    await chrome.tabs.create({ url: settingsUrl });
  };

  return (
    <div className="min-h-[600px] bg-gradient-to-b from-slate-50 to-slate-100 flex flex-col">

      <header className="border-b border-slate-200 bg-white px-5 py-4" >
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

      <div className="flex-1 overflow-y-auto px-5 py-4">
        {analysisResult && (
          <div >
            <div className={`rounded-xl px-3.5 py-3 border ${
              isSafe
                ? "bg-emerald-50 border-emerald-200/70"
                : "bg-rose-50 border-rose-200/70"
            }`}>
              <div className="flex items-start gap-2.5">
                <div className={`mt-0.5 size-5 shrink-0 ${isSafe ? "text-emerald-600" : "text-rose-600"}`}>
                  {isSafe ? (
                    <CheckCircle2 className="size-5" />
                  ) : (
                    <AlertTriangle className="size-5" />
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

            <div className="my-4">
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
              {analysisResult.domain_analysis && (
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-3 flex items-start gap-3">
                  <div className={`size-8 rounded-lg ${domainTone.iconBg} ${domainTone.iconText} flex items-center justify-center shrink-0 mt-0.5`}>
                    <Globe className="size-[14px]" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs font-semibold text-slate-900">Domain safety</p>
                      <span className={`inline-flex rounded-md ${domainTone.badgeBg} ${domainTone.badgeText} px-1.5 py-0.5 text-[11px] font-semibold`}>
                        {domainSafety ? `${domainSafety}%` : "N/A"}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] leading-4 text-slate-500 line-clamp-2">
                      {analysisResult.domain_analysis.domain || "unknown domain"}
                      {analysisResult.domain_analysis.category ? ` • ${analysisResult.domain_analysis.category}` : ""}
                    </p>
                  </div>
                </div>
              )}

              {analysisResult.image_analysis && (
                <button
                  type="button"
                  onClick={() => openDetailsPage("images")}
                  disabled={badImageItems.length === 0}
                  className={`w-full text-left rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-3 flex items-center gap-3 transition-colors ${badImageItems.length > 0 ? "cursor-pointer hover:bg-slate-100/80" : "cursor-default"}`}
                >
                  <div className="size-8 rounded-lg bg-violet-100 text-violet-700 flex items-center justify-center shrink-0">
                    <ImageIcon className="size-[14px]" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-slate-900">Image scan</p>
                    <p className="text-[11px] text-slate-500">{analysisResult.image_analysis.images_analyzed} analyzed</p>
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] shrink-0">
                    <span className="inline-flex rounded-md bg-rose-100 text-rose-700 px-1.5 py-0.5 font-semibold">{analysisResult.image_analysis.bad_images}</span>
                    <span className="inline-flex rounded-md bg-emerald-100 text-emerald-700 px-1.5 py-0.5 font-semibold">{analysisResult.image_analysis.good_images}</span>
                  </div>
                </button>
              )}

              {analysisResult.text_analysis && (
                <button
                  type="button"
                  onClick={() => openDetailsPage("text")}
                  disabled={badTextItems.length === 0}
                  className={`w-full text-left rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-3 flex items-start gap-3 transition-colors ${badTextItems.length > 0 ? "cursor-pointer hover:bg-slate-100/80" : "cursor-default"}`}
                >
                  <div className="size-8 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center shrink-0 mt-0.5">
                    <FileText className="size-[14px]" />
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
                </button>
              )}
            </div>
          </div>
        )}

        {!analysisResult && !loading && (
          <div className="h-full min-h-[220px] flex flex-col items-center justify-center text-center">
            <div className="size-12 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center mb-3">
              <Shield className="size-5 text-slate-500" strokeWidth={1.8} />
            </div>
            <p className="text-sm font-semibold text-slate-900">Ready to scan this page</p>
            <p className="mt-1 text-xs text-slate-500">Run verification to see a full safety report.</p>
          </div>
        )}

        {loading && !analysisResult && (
          <div className="h-full min-h-[220px] flex flex-col items-center justify-center text-center">
            <div className="size-10 mb-3">
              <Loader2 className="size-10 animate-spin text-indigo-600" />
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
            className="inline-flex min-w-[230px] items-center justify-center rounded-2xl bg-indigo-600 px-10 py-2 text-sm font-semibold tracking-tight text-white shadow-[0_10px_20px_rgba(79,70,229,0.28)] hover:bg-indigo-500 transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:bg-indigo-400 disabled:cursor-not-allowed"
          >
            {loading ? "Scanning…" : "Run Verification"}
          </button>
        </div>

        <div className="mt-5 flex justify-center">
          <button
            onClick={sendHeartBeatEvent}
            disabled={loading}
            className="inline-flex items-center rounded-lg px-3 py-2 text-xs font-medium text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <HeartPlus className="size-4 mr-1" />Send Heartbeat
          </button>
                    <button
            onClick={openSettingsPopup}
            disabled={loading}
            className="inline-flex items-center rounded-lg px-3 py-2 text-xs font-medium text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Settings className="size-4 mr-1" />Settings
          </button>
        </div>
      </div>
    </div>
  );
}

export default Popup;
