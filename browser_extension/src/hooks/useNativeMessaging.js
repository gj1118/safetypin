// src/hooks/useNativeMessaging.js

import { useState, useEffect, useCallback } from "react";

export function useNativeMessaging() {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const hasChromeRuntime = typeof chrome !== "undefined" && !!chrome.runtime?.sendMessage;
  const hasChromeOnMessage = typeof chrome !== "undefined" && !!chrome.runtime?.onMessage;

  // Connect to native host
  const connect = useCallback(async () => {
    if (!hasChromeRuntime) {
      setIsConnected(false);
      return { success: false, error: "chrome_runtime_unavailable" };
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: "connectNative",
      });

      setIsConnected(response.success);
      return response;
    } catch (error) {
      console.error("Failed to connect:", error);
      return { success: false, error: error.message };
    }
  }, [hasChromeRuntime]);

  // Disconnect from native host
  const disconnect = useCallback(async () => {
    if (!hasChromeRuntime) {
      setIsConnected(false);
      return { success: false, error: "chrome_runtime_unavailable" };
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: "disconnectNative",
      });

      setIsConnected(false);
      return response;
    } catch (error) {
      console.error("Failed to disconnect:", error);
      return { success: false, error: error.message };
    }
  }, [hasChromeRuntime]);

  // Send message to native host
  const sendMessage = useCallback(async (data) => {
    if (!hasChromeRuntime) {
      return { success: false, error: "chrome_runtime_unavailable" };
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: "sendToNative",
        data,
      });

      return response;
    } catch (error) {
      console.error("Failed to send message:", error);
      return { success: false, error: error.message };
    }
  }, [hasChromeRuntime]);

  // Check connection status
  const checkStatus = useCallback(async () => {
    if (!hasChromeRuntime) {
      setIsConnected(false);
      return { isConnected: false };
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: "getConnectionStatus",
      });

      setIsConnected(response.isConnected);
      return response;
    } catch (error) {
      console.error("Failed to check status:", error);
      return { isConnected: false };
    }
  }, [hasChromeRuntime]);

  // Analyze DOM from a specific tab
  const analyzeDom = useCallback(async (tabId) => {
    if (!hasChromeRuntime) {
      return { success: false, error: "chrome_runtime_unavailable" };
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: "analyzeDom",
        tabId,
      });

      return response;
    } catch (error) {
      console.error("Failed to analyze DOM:", error);
      return { success: false, error: error.message };
    }
  }, [hasChromeRuntime]);

  // Listen for messages from background script
  useEffect(() => {
    if (!hasChromeOnMessage) {
      setIsConnected(false);
      return;
    }

    const handleMessage = (message) => {
      if (message.type === "nativeMessage") {
        setMessages((prev) => [...prev, message.data]);
      } else if (message.type === "nativeDisconnected") {
        setIsConnected(false);
      }
    };

    chrome.runtime.onMessage.addListener(handleMessage);

    // Check initial status
    checkStatus();

    return () => {
      chrome.runtime.onMessage.removeListener(handleMessage);
    };
  }, [checkStatus, hasChromeOnMessage]);

  return {
    isConnected,
    messages,
    connect,
    disconnect,
    sendMessage,
    checkStatus,
    analyzeDom,
  };
}
