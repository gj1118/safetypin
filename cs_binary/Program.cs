using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Mail;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Data.Sqlite;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using HtmlAgilityPack;

namespace CsBinary;

public class Program
{
    private const int MaxMessageSize = 1024 * 1024; // 1MB
    private const int MaxTextLength = 256;
    private const int DomImageParallelism = 4;
    private const double LowReputationSafetyThreshold = 0.70;
    private const double NotificationSafetyThreshold = 0.80;
    private static readonly TimeSpan NotificationCooldown = TimeSpan.FromMinutes(5);
    
    private static InferenceSession? _imageSession;
    private static InferenceSession? _textSession;
    private static InferenceSession? _domainSession;
    private static Dictionary<int, string> _reasonNames = new();
    private static Dictionary<string, int> _vocabulary = new();
    private static Dictionary<string, string> _domainReputation = new();
    private static Dictionary<char, int> _charToIdx = new();
    private static int _domainMaxLength = 64;
    private static bool _modelsLoaded;
    private static double? _currentMessageId;
    private const int VocabUnknown = 100;
    private const int VocabCls = 101;
    private const int VocabSep = 102;
    private const int VocabPad = 0;
    private const int VocabMask = 103;

    private static readonly HttpClient _httpClient = new HttpClient
    {
        Timeout = TimeSpan.FromSeconds(10)
    };

    private static readonly object _dbLock = new object();
    private static string _sqliteDbPath = string.Empty;
    private static readonly ConcurrentDictionary<string, long> _lastNotificationByKey = new();

    public static int Main(string[] args)
    {
        AppLogger.Initialize();
        Console.Error.WriteLine("[INFO] C# Native Messaging Host starting...");

        try
        {
            EnsureDatabaseInitialized();
            LoadModels();
            
            if (args.Length > 0 && args[0] == "--install")
            {
                InstallNativeHost(args.Skip(1).ToArray());
                return 0;
            }

            RunMessagingHost();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Fatal error: {ex.Message}");
            return 1;
        }
        finally
        {
            AppLogger.Shutdown();
        }
    }

    private static void LoadModels()
    {
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var modelsDir = Path.Combine(baseDir, "Models");

        Console.Error.WriteLine("[INFO] Loading models from: " + modelsDir);

        try
        {
            var imageModelPath = Path.Combine(modelsDir, "image_classifier.onnx");
            if (File.Exists(imageModelPath))
            {
                var sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _imageSession = new InferenceSession(imageModelPath, sessionOptions);
                Console.Error.WriteLine("[INFO] Image classifier loaded successfully");
            }
            else
            {
                Console.Error.WriteLine("[WARNING] Image model not found at: " + imageModelPath);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to load image classifier: {ex.Message}");
        }

        try
        {
            var textModelPath = Path.Combine(modelsDir, "text_classifier.onnx");
            if (File.Exists(textModelPath))
            {
                var sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _textSession = new InferenceSession(textModelPath, sessionOptions);

                var reasonPath = Path.Combine(modelsDir, "bert_text_with_reasons", "reason_categories.json");
                if (File.Exists(reasonPath))
                {
                    var reasonJson = File.ReadAllText(reasonPath);
                    var reasonData = JsonSerializer.Deserialize<Dictionary<string, int>>(reasonJson);
                    if (reasonData != null)
                    {
                        _reasonNames = reasonData.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
                    }
                }

                Console.Error.WriteLine("[INFO] Text classifier loaded successfully");
            }
            else
            {
                Console.Error.WriteLine("[WARNING] Text model not found at: " + textModelPath);
            }

            try
            {
                var vocabPath = Path.Combine(modelsDir, "bert_text_with_reasons", "vocab.json");
                if (File.Exists(vocabPath))
                {
                    var vocabJson = File.ReadAllText(vocabPath);
                    var vocabData = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson);
                    if (vocabData != null)
                    {
                        _vocabulary = vocabData;
                        Console.Error.WriteLine($"[INFO] Loaded vocabulary with {_vocabulary.Count} tokens");
                    }
                }
                else
                {
                    Console.Error.WriteLine("[WARNING] Vocabulary not found at: " + vocabPath);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Failed to load vocabulary: {ex.Message}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to load text classifier: {ex.Message}");
        }

        _modelsLoaded = _textSession != null || _imageSession != null;
        Console.Error.WriteLine($"[INFO] Models loaded: Image={_imageSession != null}, Text={_textSession != null}");

        // Load domain reputation
        try
        {
            var domainRepPath = Path.Combine(modelsDir, "domain_reputation.json");
            if (File.Exists(domainRepPath))
            {
                var domainJson = File.ReadAllText(domainRepPath);
                var domainData = JsonSerializer.Deserialize<Dictionary<string, string>>(domainJson);
                if (domainData != null)
                {
                    var normalizedDomainData = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                    foreach (var kvp in domainData)
                    {
                        var normalizedDomain = NormalizeDomain(kvp.Key);
                        if (!string.IsNullOrEmpty(normalizedDomain))
                        {
                            normalizedDomainData[normalizedDomain] = kvp.Value;
                        }
                    }

                    _domainReputation = normalizedDomainData;
                    Console.Error.WriteLine($"[INFO] Loaded domain reputation with {_domainReputation.Count} domains");
                }
            }
            else
            {
                Console.Error.WriteLine("[WARNING] Domain reputation file not found");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to load domain reputation: {ex.Message}");
        }

        // Load domain classifier ONNX model
        try
        {
            var domainModelPath = Path.Combine(modelsDir, "text_classifier_domain.onnx");
            if (File.Exists(domainModelPath))
            {
                var sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _domainSession = new InferenceSession(domainModelPath, sessionOptions);
                Console.Error.WriteLine("[INFO] Domain classifier loaded successfully");

                // Load metadata for domain classifier
                var metadataPath = Path.Combine(modelsDir, "domain_classifier_cnn", "metadata.json");
                if (File.Exists(metadataPath))
                {
                    var metadataJson = File.ReadAllText(metadataPath);
                    using var doc = JsonDocument.Parse(metadataJson);
                    _domainMaxLength = doc.RootElement.GetProperty("max_length").GetInt32();
                    
                    _charToIdx = new Dictionary<char, int>();
                    var charToIdxProp = doc.RootElement.GetProperty("char_to_idx");
                    foreach (var prop in charToIdxProp.EnumerateObject())
                    {
                        _charToIdx[prop.Name[0]] = prop.Value.GetInt32();
                    }
                    Console.Error.WriteLine($"[INFO] Domain classifier metadata loaded: max_length={_domainMaxLength}, vocab_size={_charToIdx.Count}");
                }
            }
            else
            {
                Console.Error.WriteLine("[WARNING] Domain classifier model not found");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to load domain classifier: {ex.Message}");
        }
    }

    private static void RunMessagingHost()
    {
        Console.Error.WriteLine("[INFO] Native messaging host started");
        Console.Error.WriteLine("[INFO] Communication mode: STDIN/STDOUT");
        Console.Error.WriteLine($"[INFO] Max message size: {MaxMessageSize} bytes");

        var stdin = Console.OpenStandardInput();
        var stdout = Console.OpenStandardOutput();

        while (true)
        {
            try
            {
                var message = ReadMessage(stdin);
                if (message == null)
                {
                    Console.Error.WriteLine("[INFO] Extension disconnected (EOF)");
                    break;
                }

                var response = HandleMessage(message);
                SendMessage(stdout, response);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Error processing message: {ex.Message}");
                var errorResponse = FormatError("internal_error", ex.Message);
                try { SendMessage(stdout, errorResponse); } catch { }
            }
            finally
            {
                AppLogger.ClearRequestScope();
            }
        }

        Console.Error.WriteLine("[INFO] Native messaging host stopped");
    }

    private static byte[]? ReadMessage(Stream stdin)
    {
        var lengthBytes = new byte[4];
        var bytesRead = stdin.Read(lengthBytes, 0, 4);
        
        if (bytesRead == 0)
            return null;

        if (bytesRead != 4)
        {
            Console.Error.WriteLine("[ERROR] Incomplete length header");
            throw new InvalidOperationException("Invalid message header");
        }

        var messageLength = BitConverter.ToUInt32(lengthBytes, 0);
        
        if (messageLength == 0)
        {
            Console.Error.WriteLine("[ERROR] Zero-length message");
            throw new InvalidOperationException("Invalid message length");
        }

        if (messageLength > MaxMessageSize)
        {
            Console.Error.WriteLine("[ERROR] Message too large");
            throw new InvalidOperationException("Message too large");
        }

        var message = new byte[messageLength];
        var totalRead = 0;
        while (totalRead < messageLength)
        {
            var read = stdin.Read(message, totalRead, (int)(messageLength - totalRead));
            if (read == 0)
                throw new InvalidOperationException("Incomplete message");
            totalRead += read;
        }

        return message;
    }

    private static void SendMessage(Stream stdout, byte[] message)
    {
        if (message.Length > MaxMessageSize)
            throw new InvalidOperationException("Message too large");

        var lengthBytes = BitConverter.GetBytes((uint)message.Length);
        stdout.Write(lengthBytes, 0, 4);
        stdout.Write(message, 0, message.Length);
        stdout.Flush();

        Console.Error.WriteLine($"[DEBUG] Sent message: {message.Length} bytes");
    }

    private static byte[] HandleMessage(byte[] messageJson)
    {
        var messageStopwatch = Stopwatch.StartNew();
        var message = Encoding.UTF8.GetString(messageJson);
        Console.Error.WriteLine($"[TRACE] Received message JSON preview: {TruncateForLog(message, 300)}");
        
        JsonDocument? doc = null;
        JsonElement root;
        double? messageId = null;
        
        try
        {
            doc = JsonDocument.Parse(message);
            root = doc.RootElement;
            
            // Extract messageId from the outer message
            if (root.TryGetProperty("messageId", out var midElement))
            {
                try {
                    if (midElement.ValueKind == JsonValueKind.Number)
                    {
                        messageId = midElement.GetDouble();
                        _currentMessageId = messageId;
                    }
                    else if (midElement.ValueKind == JsonValueKind.String && double.TryParse(midElement.GetString(), out var parsedId))
                    {
                        messageId = parsedId;
                        _currentMessageId = messageId;
                    }
                } catch (Exception ex) {
                    Console.Error.WriteLine($"[ERROR] Failed to parse messageId: {ex.Message}");
                }
            }
        }
        catch (JsonException ex)
        {
            Console.Error.WriteLine($"[ERROR] JSON parse error: {ex.Message}");
            return FormatError("invalid_json", ex.Message, messageId);
        }

        // Handle wrapped message from extension: {type: "sendToNative", data: {...}}
        if (root.TryGetProperty("type", out var outerTypeElement))
        {
            var outerType = outerTypeElement.GetString() ?? "";
            if (outerType == "sendToNative" && root.TryGetProperty("data", out var dataElement))
            {
                root = dataElement;
            }
        }

        // Also check if message is wrapped in "data" field directly
        if (root.TryGetProperty("data", out var dataElement2))
        {
            root = dataElement2;
        }

        var msgType = "unknown";
        if (root.TryGetProperty("type", out var typeElement2))
        {
            msgType = typeElement2.GetString() ?? "unknown";
        }

        var requestScopeId = BuildRequestScopeId(messageId, msgType);
        AppLogger.SetRequestScope(requestScopeId);
        Console.Error.WriteLine($"[DEBUG] Received message: {messageJson.Length} bytes");

        if (msgType == "unknown")
        {
            Console.Error.WriteLine("[WARNING] Missing 'type' field in incoming message");
            return FormatError("missing_field", "Missing 'type' field", messageId);
        }

        Console.Error.WriteLine($"[DEBUG] Message type: {msgType}, messageId: {messageId}");

        try
        {
            var responseBytes = msgType switch
            {
                "content_analysis" => HandleContentAnalysis(root),
                "url_analysis" => HandleUrlAnalysis(root),
                "dom_analysis" => HandleDomAnalysis(root),
                "heartbeat" => HandleHeartbeat(),
                "get_status" => HandleGetStatus(),
                "get_user_info" => HandleGetUserInfo(),
                "get_low_reputation_history" => HandleGetLowReputationHistory(root),
                "save_user_info" => HandleSaveUserInfo(root),
                "classify_text" => HandleClassifyText(root),
                "classify_image" => HandleClassifyImage(root),
                "classify_both" => HandleClassifyBoth(root),
                _ => FormatError("unknown_type", msgType, messageId)
            };

            Console.Error.WriteLine($"[DEBUG] Message handled: type={msgType}, elapsed_ms={messageStopwatch.ElapsedMilliseconds}, response_bytes={responseBytes.Length}");
            return responseBytes;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Handle message error: {ex.Message}");
            Console.Error.WriteLine($"[ERROR] Stack trace: {ex.StackTrace}");
            return FormatError("handler_error", ex.Message, messageId);
        }
        finally
        {
            doc?.Dispose();
        }
    }

    private static byte[] HandleContentAnalysis(JsonElement root)
    {
        string? text = null;
        string? imageBase64 = null;

        if (root.TryGetProperty("text", out var textElement))
            text = textElement.GetString();
        
        if (root.TryGetProperty("image", out var imageElement))
            imageBase64 = imageElement.GetString();

        long? requestId = null;
        if (root.TryGetProperty("requestId", out var idElement))
        {
            try {
                if (idElement.ValueKind == JsonValueKind.Number)
                {
                    requestId = idElement.GetInt64();
                }
                else if (idElement.ValueKind == JsonValueKind.String && long.TryParse(idElement.GetString(), out var parsedId))
                {
                    requestId = parsedId;
                }
            } catch { }
        }

        var result = new Dictionary<string, object>();
        
        if (!string.IsNullOrEmpty(text) && _textSession != null)
        {
            var textResult = ClassifyText(text);
            result["text_classification"] = textResult.Classification;
            result["text_confidence"] = textResult.Confidence;
            result["text_reason"] = textResult.Reason;
        }

        if (!string.IsNullOrEmpty(imageBase64) && _imageSession != null)
        {
            var imageResult = ClassifyImage(imageBase64);
            result["image_classification"] = imageResult.Classification;
            result["image_confidence"] = imageResult.Confidence;
        }

        var finalClassification = "good";
        double finalConfidence = 0.5;
        
        if (result.Count > 0)
        {
            if (result.ContainsKey("text_classification") && result["text_classification"]?.ToString() == "bad")
            {
                finalClassification = "bad";
                finalConfidence = Convert.ToDouble(result["text_confidence"]);
            }
            if (result.ContainsKey("image_classification") && result["image_classification"]?.ToString() == "bad")
            {
                finalClassification = "bad";
                finalConfidence = Math.Max(finalConfidence, Convert.ToDouble(result["image_confidence"]));
            }
        }

        var response = new Dictionary<string, object>
        {
            ["classification"] = finalClassification,
            ["confidence"] = finalConfidence,
            ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        };

        if (requestId.HasValue)
            response["requestId"] = requestId.Value;

        foreach (var kvp in result)
            response[kvp.Key] = kvp.Value;

        var json = JsonSerializer.Serialize(response);
        return Encoding.UTF8.GetBytes(json);
    }

    private static byte[] HandleUrlAnalysis(JsonElement root)
    {
        if (!root.TryGetProperty("url", out var urlElement))
        {
            return FormatError("missing_field", "Missing 'url' field");
        }

        var url = urlElement.GetString() ?? "";

        if (string.IsNullOrEmpty(url))
        {
            return FormatError("invalid_url", "URL is empty");
        }

        long? requestId = null;
        if (root.TryGetProperty("requestId", out var idElement))
        {
            try {
                if (idElement.ValueKind == JsonValueKind.Number)
                {
                    requestId = idElement.GetInt64();
                }
                else if (idElement.ValueKind == JsonValueKind.String && long.TryParse(idElement.GetString(), out var parsedId))
                {
                    requestId = parsedId;
                }
            } catch { }
        }

        try
        {
            var uri = new Uri(url);
            var domain = uri.Host;

            Console.Error.WriteLine($"[INFO] Analyzing URL: {url}");

            // Check domain reputation
            string reputation = "unknown";
            bool inDatabase = false;
            double riskScore = ClassifyDomain(domain);

            if (_domainReputation.Count > 0 && TryGetDomainReputation(domain, out reputation))
            {
                inDatabase = true;
            }

            string htmlContent = "";
            string textContent = "";
            
            try
            {
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                request.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
                var httpResponse = _httpClient.Send(request);
                httpResponse.EnsureSuccessStatusCode();
                htmlContent = httpResponse.Content.ReadAsStringAsync().Result;

                var doc = new HtmlDocument();
                doc.LoadHtml(htmlContent);
                textContent = doc.DocumentNode.InnerText ?? "";
                textContent = Regex.Replace(textContent, @"\s+", " ").Trim();
                if (textContent.Length > 2000)
                    textContent = textContent.Substring(0, 2000);

                Console.Error.WriteLine($"[INFO] Downloaded HTML, extracted {textContent.Length} chars of text");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Failed to fetch URL: {ex.Message}");
            }

            var textAnalysis = new Dictionary<string, object>
            {
                ["classification"] = "good",
                ["confidence"] = 0.5,
                ["reason"] = "no_text_analyzed"
            };

            var textElementResults = new List<object>();
            
            if (!string.IsNullOrEmpty(textContent) && _textSession != null)
            {
                try
                {
                    var doc = new HtmlDocument();
                    doc.LoadHtml(htmlContent);
                    
                    var textSelectors = new[] { "//p", "//h1", "//h2", "//h3", "//h4", "//h5", "//h6", "//span", "//div", "//a", "//li", "//td", "//th" };
                    
                    foreach (var selector in textSelectors)
                    {
                        var nodes = doc.DocumentNode.SelectNodes(selector);
                        if (nodes != null)
                        {
                            foreach (var node in nodes.Take(20))
                            {
                                var elementText = node.InnerText?.Trim() ?? "";
                                if (string.IsNullOrEmpty(elementText) || elementText.Length < 10)
                                    continue;
                                
                                elementText = Regex.Replace(elementText, @"\s+", " ").Trim();
                                if (elementText.Length > 500)
                                    elementText = elementText.Substring(0, 500);
                                
                                try
                                {
                                    var result = ClassifyText(elementText);
                                    if (result.Classification == "bad")
                                    {
                                        textElementResults.Add(new Dictionary<string, object>
                                        {
                                            ["text"] = elementText.Length > 100 ? elementText.Substring(0, 100) + "..." : elementText,
                                            ["classification"] = result.Classification,
                                            ["confidence"] = result.Confidence,
                                            ["reason"] = result.Reason,
                                            ["element_info"] = GetElementInfo(node)
                                        });
                                    }
                                }
                                catch { }
                            }
                        }
                    }

                    var overallResult = ClassifyText(textContent);
                    textAnalysis["classification"] = overallResult.Classification;
                    textAnalysis["confidence"] = overallResult.Confidence;
                    textAnalysis["reason"] = overallResult.Reason;
                    textAnalysis["bad_text_elements"] = textElementResults;
                    
                    Console.Error.WriteLine($"[INFO] Text classified: {overallResult.Classification} ({overallResult.Confidence:P2}), bad elements: {textElementResults.Count}");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[ERROR] Text classification failed: {ex.Message}");
                }
            }

            var imageAnalysis = new Dictionary<string, object>
            {
                ["has_harmful_images"] = false,
                ["images_analyzed"] = 0,
                ["bad_images"] = 0,
                ["good_images"] = 0,
                ["image_results"] = new object[0]
            };

            if (!string.IsNullOrEmpty(htmlContent) && _imageSession != null)
            {
                try
                {
                    var doc = new HtmlDocument();
                    doc.LoadHtml(htmlContent);
                    
                    var imageInfoList = new List<Tuple<string, HtmlNode>>();

                    var imgNodes = doc.DocumentNode.SelectNodes("//img[@src]");
                    if (imgNodes != null)
                    {
                        foreach (var imgNode in imgNodes)
                        {
                            var src = imgNode.GetAttributeValue("src", "");
                            var dataSrc = imgNode.GetAttributeValue("data-src", "");
                            var dataLazySrc = imgNode.GetAttributeValue("data-lazy-src", "");
                            var dataSrcset = imgNode.GetAttributeValue("data-srcset", "");
                            
                            if (!string.IsNullOrEmpty(src)) imageInfoList.Add(Tuple.Create(src, imgNode));
                            if (!string.IsNullOrEmpty(dataSrc)) imageInfoList.Add(Tuple.Create(dataSrc, imgNode));
                            if (!string.IsNullOrEmpty(dataLazySrc)) imageInfoList.Add(Tuple.Create(dataLazySrc, imgNode));
                            if (!string.IsNullOrEmpty(dataSrcset))
                            {
                                var parts = dataSrcset.Split(',');
                                foreach (var part in parts)
                                {
                                    var urlPart = part.Trim().Split(' ')[0];
                                    if (!string.IsNullOrEmpty(urlPart))
                                        imageInfoList.Add(Tuple.Create(urlPart, imgNode));
                                }
                            }
                        }
                    }

                    var sourceNodes = doc.DocumentNode.SelectNodes("//source[@srcset]");
                    if (sourceNodes != null)
                    {
                        foreach (var sourceNode in sourceNodes)
                        {
                            var srcset = sourceNode.GetAttributeValue("srcset", "");
                            if (!string.IsNullOrEmpty(srcset))
                            {
                                var parts = srcset.Split(',');
                                foreach (var part in parts)
                                {
                                    var urlPart = part.Trim().Split(' ')[0];
                                    if (!string.IsNullOrEmpty(urlPart))
                                        imageInfoList.Add(Tuple.Create(urlPart, sourceNode));
                                }
                            }
                        }
                    }

                    var pictureNodes = doc.DocumentNode.SelectNodes("//picture/img[@src]");
                    if (pictureNodes != null)
                    {
                        foreach (var picImg in pictureNodes)
                        {
                            var src = picImg.GetAttributeValue("src", "");
                            if (!string.IsNullOrEmpty(src))
                                imageInfoList.Add(Tuple.Create(src, picImg));
                        }
                    }

                    var analyzedImages = new List<object>();
                    int goodImages = 0;
                    int badImages = 0;

                    Console.Error.WriteLine($"[INFO] Found {imageInfoList.Count} image URLs to analyze");

                    var processedUrls = new HashSet<string>();
                    foreach (var imgInfo in imageInfoList.Take(20))
                    {
                        var rawImgUrl = imgInfo.Item1;
                        var imgNode = imgInfo.Item2;

                        if (string.IsNullOrEmpty(rawImgUrl))
                            continue;

                        if (processedUrls.Contains(rawImgUrl))
                            continue;
                        processedUrls.Add(rawImgUrl);

                        if (string.IsNullOrEmpty(rawImgUrl))
                            continue;

                        var imgUrl = rawImgUrl;
                        if (imgUrl.StartsWith("//"))
                            imgUrl = "https:" + imgUrl;
                        else if (imgUrl.StartsWith("/"))
                            imgUrl = new Uri(new Uri(url), imgUrl).ToString();
                        else if (!imgUrl.StartsWith("http"))
                            imgUrl = new Uri(new Uri(url), imgUrl).ToString();

                        if (imgUrl.EndsWith(".svg", StringComparison.OrdinalIgnoreCase) ||
                            imgUrl.EndsWith(".ico", StringComparison.OrdinalIgnoreCase) ||
                            imgUrl.EndsWith(".gif", StringComparison.OrdinalIgnoreCase) ||
                            imgUrl.EndsWith(".webp", StringComparison.OrdinalIgnoreCase))
                        {
                            continue;
                        }

                        try
                        {
                            using var imgRequest = new HttpRequestMessage(HttpMethod.Get, imgUrl);
                            imgRequest.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
                            using var imgResponse = _httpClient.Send(imgRequest);
                            if (imgResponse.IsSuccessStatusCode)
                            {
                                var imgBytes = imgResponse.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();
                                var contentType = imgResponse.Content.Headers.ContentType?.MediaType ?? "unknown";
                                var imgResult = ClassifyImage(imgBytes);
                                
                                var elementInfo = GetElementInfo(imgNode);
                                
                                analyzedImages.Add(new Dictionary<string, object>
                                {
                                    ["image_url"] = imgUrl,
                                    ["classification"] = imgResult.Classification,
                                    ["confidence"] = imgResult.Confidence,
                                    ["image_size"] = imgBytes.Length,
                                    ["content_type"] = contentType,
                                    ["element_info"] = elementInfo
                                });

                                if (imgResult.Classification == "good")
                                    goodImages++;
                                else
                                    badImages++;

                                Console.Error.WriteLine($"[INFO] Image classified: {imgUrl} -> {imgResult.Classification} ({imgResult.Confidence:P2})");
                            }
                        }
                        catch
                        {
                        }
                    }

                    var badImageElements = new List<object>();
                    foreach (var img in analyzedImages)
                    {
                        var imgDict = (Dictionary<string, object>)img;
                        if (imgDict["classification"]?.ToString() == "bad")
                        {
                            badImageElements.Add(new Dictionary<string, object>
                            {
                                ["url"] = imgDict["image_url"],
                                ["confidence"] = imgDict["confidence"],
                                ["classification"] = imgDict["classification"],
                                ["element_info"] = imgDict["element_info"]
                            });
                        }
                    }

                    imageAnalysis["has_harmful_images"] = badImages > 0;
                    imageAnalysis["images_analyzed"] = analyzedImages.Count;
                    imageAnalysis["bad_images"] = badImages;
                    imageAnalysis["good_images"] = goodImages;
                    imageAnalysis["image_results"] = analyzedImages;
                    imageAnalysis["bad_image_elements"] = badImageElements;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[ERROR] Image analysis failed: {ex.Message}");
                }
            }

            var finalClassification = "good";
            var finalConfidence = 0.5;
            var finalReasons = new List<string>();

            if (textAnalysis["classification"]?.ToString() == "bad")
            {
                finalClassification = "bad";
                finalConfidence = Convert.ToDouble(textAnalysis["confidence"]);
                finalReasons.Add(textAnalysis["reason"]?.ToString() ?? "unknown");
            }

            if (Convert.ToBoolean(imageAnalysis["has_harmful_images"]))
            {
                finalClassification = "bad";
                finalConfidence = 0.95;
                finalReasons.Clear();
                finalReasons.Add("harmful_images_detected");
            }

            var response = new Dictionary<string, object>
            {
                ["page_classification"] = finalClassification,
                ["page_confidence"] = finalConfidence,
                ["page_reasons"] = finalReasons.ToArray(),
                ["text_analysis"] = textAnalysis,
                ["domain_analysis"] = new Dictionary<string, object>
                {
                    ["domain"] = domain,
                    ["risk_score"] = riskScore,
                    ["category"] = reputation,
                    ["in_database"] = inDatabase
                },
                ["image_analysis"] = imageAnalysis,
                ["type"] = "url",
                ["requested_url"] = url,
                ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };

            if (requestId.HasValue)
                response["requestId"] = requestId.Value;

            RecordLowReputationVisitIfNeeded(
                url,
                domain,
                riskScore,
                "url_analysis",
                finalClassification,
                finalReasons,
                finalClassification == "bad" ? "page_blocked" : "content_removed");

            return SerializeResponse(response);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] URL analysis error: {ex.Message}");
            return FormatError("url_analysis_error", ex.Message);
        }
    }

    private static byte[] HandleDomAnalysis(JsonElement root)
    {
        var domStopwatch = Stopwatch.StartNew();
        string? url = null;
        string? html = null;
        List<object>? images = null;
        List<object>? textElements = null;

        if (root.TryGetProperty("url", out var urlElement))
            url = urlElement.GetString();
        
        if (root.TryGetProperty("html", out var htmlElement))
            html = htmlElement.GetString();
        
        if (root.TryGetProperty("images", out var imagesElement))
        {
            images = new List<object>();
            foreach (var img in imagesElement.EnumerateArray())
            {
                var imgDict = new Dictionary<string, object>();
                if (img.TryGetProperty("src", out var src))
                    imgDict["src"] = src.GetString() ?? "";
                if (img.TryGetProperty("tag", out var tag))
                    imgDict["tag"] = tag.GetString() ?? "img";
                if (img.TryGetProperty("id", out var id))
                    imgDict["id"] = id.GetString() ?? "";
                if (img.TryGetProperty("class", out var cls))
                    imgDict["class"] = cls.GetString() ?? "";
                if (img.TryGetProperty("alt", out var alt))
                    imgDict["alt"] = alt.GetString() ?? "";
                if (img.TryGetProperty("idx", out var idx))
                    imgDict["idx"] = idx.GetInt32();
                images.Add(imgDict);
            }
        }
        
        if (root.TryGetProperty("textElements", out var textElElement))
        {
            textElements = new List<object>();
            foreach (var el in textElElement.EnumerateArray())
            {
                var textDict = new Dictionary<string, object>();
                if (el.TryGetProperty("text", out var text))
                    textDict["text"] = text.GetString() ?? "";
                if (el.TryGetProperty("tag", out var tag))
                    textDict["tag"] = tag.GetString() ?? "div";
                if (el.TryGetProperty("id", out var id))
                    textDict["id"] = id.GetString() ?? "";
                if (el.TryGetProperty("class", out var cls))
                    textDict["class"] = cls.GetString() ?? "";
                textElements.Add(textDict);
            }
        }

        long? requestId = null;
        if (root.TryGetProperty("requestId", out var idElement))
        {
            try { requestId = idElement.GetInt64(); } catch { }
        }

        try
        {
            Console.Error.WriteLine($"[INFO] Analyzing DOM from extension. URL: {url}, Images: {images?.Count ?? 0}, Text elements: {textElements?.Count ?? 0}");

            bool foundBad = false;
            double highestConfidence = 0.0;
            var textAnalysisStopwatch = Stopwatch.StartNew();

            var textAnalysis = new Dictionary<string, object>
            {
                ["classification"] = "good",
                ["confidence"] = 0.5,
                ["reason"] = "no_text_analyzed"
            };

            var textElementResults = new List<object>();
            if (textElements != null && _textSession != null)
            {
                foreach (var te in textElements)
                {
                    var teDict = (Dictionary<string, object>)te;
                    var text = teDict["text"]?.ToString() ?? "";
                    
                    if (string.IsNullOrEmpty(text) || text.Length < 20)
                        continue;

                    try
                    {
                        var result = ClassifyText(text);
                        if (result.Classification == "bad" && result.Confidence > 0.85)
                        {
                            if (result.Confidence > highestConfidence)
                                highestConfidence = result.Confidence;
                            
                            textElementResults.Add(new Dictionary<string, object>
                            {
                                ["text"] = text.Length > 100 ? text.Substring(0, 100) + "..." : text,
                                ["classification"] = result.Classification,
                                ["confidence"] = result.Confidence,
                                ["reason"] = result.Reason,
                                ["element_info"] = new Dictionary<string, object>
                                {
                                    ["tag"] = teDict["tag"] ?? "div",
                                    ["id"] = teDict["id"] ?? "",
                                    ["class"] = teDict["class"] ?? "",
                                    ["css_selector"] = BuildCssSelector(teDict["tag"]?.ToString() ?? "div", teDict["id"]?.ToString() ?? "", teDict["class"]?.ToString() ?? "")
                                }
                            });
                            
                            // Return immediately when bad content found with > 85% confidence
                            Console.Error.WriteLine($"[INFO] Bad text found ({result.Confidence:P0}), returning early");
                            textAnalysis["classification"] = "bad";
                            textAnalysis["confidence"] = result.Confidence;
                            textAnalysis["reason"] = result.Reason;
                            textAnalysis["bad_text_elements"] = textElementResults;
                            foundBad = true;
                            break;
                        }
                    }
                    catch { }
                }

                if (!foundBad)
                {
                    textAnalysis["bad_text_elements"] = textElementResults;
                }
                Console.Error.WriteLine($"[INFO] Text analysis done. Bad elements: {textElementResults.Count}");
            }

            textAnalysisStopwatch.Stop();
            Console.Error.WriteLine($"[DEBUG] Text analysis duration_ms={textAnalysisStopwatch.ElapsedMilliseconds}");

            // Get domain risk score BEFORE the early return check
            double domainRiskForText = 0.5;
            string domainStrForText = "";
            if (!string.IsNullOrEmpty(url))
            {
                try
                {
                    var uri = new Uri(url);
                    domainStrForText = uri.Host;
                    domainRiskForText = ClassifyDomain(domainStrForText);
                }
                catch { }
            }

            // For high reputation domains, don't block on text false positives
            // Only return early for high-risk domains or harmful images
            if (foundBad && domainRiskForText >= 0.9)
            {
                var earlyResponse = new Dictionary<string, object>
                {
                    ["page_classification"] = "bad",
                    ["page_confidence"] = highestConfidence,
                    ["page_reasons"] = new[] { "harmful_text_detected" },
                    ["action_taken"] = "page_blocked",
                    ["text_analysis"] = textAnalysis,
                    ["image_analysis"] = new Dictionary<string, object>
                    {
                        ["has_harmful_images"] = false,
                        ["images_analyzed"] = 0,
                        ["bad_images"] = 0,
                        ["good_images"] = 0,
                        ["bad_image_elements"] = new List<object>()
                    },
                    ["domain_analysis"] = new Dictionary<string, object>
                    {
                        ["domain"] = domainStrForText,
                        ["risk_score"] = domainRiskForText,
                        ["category"] = "unknown",
                        ["in_database"] = false
                    },
                    ["type"] = "dom",
                    ["requested_url"] = url ?? "",
                    ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
                };

                if (requestId.HasValue)
                    earlyResponse["requestId"] = requestId.Value;

                return SerializeResponse(earlyResponse);
            }

            var imageAnalysis = new Dictionary<string, object>
            {
                ["has_harmful_images"] = false,
                ["images_analyzed"] = 0,
                ["bad_images"] = 0,
                ["good_images"] = 0
            };

            var badImageElements = new List<object>();
            double maxBadImageConfidence = 0.0;
            var imageAnalysisStopwatch = Stopwatch.StartNew();

            if (images != null && _imageSession != null)
            {
                int analyzedCount = 0;
                var processedImageUrls = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var uniqueImageDicts = new List<Dictionary<string, object>>();

                foreach (var imgInfo in images)
                {
                    var imgDict = (Dictionary<string, object>)imgInfo;
                    var imgUrl = imgDict["src"]?.ToString() ?? "";

                    if (string.IsNullOrEmpty(imgUrl) || imgUrl.StartsWith("data:"))
                        continue;

                    if (!processedImageUrls.Add(imgUrl))
                        continue;

                    uniqueImageDicts.Add(imgDict);
                }

                Console.Error.WriteLine($"[DEBUG] DOM image analysis: unique_images={uniqueImageDicts.Count}, parallelism={DomImageParallelism}");

                var badImageBag = new ConcurrentBag<object>();
                var confidenceLock = new object();
                var parallelOptions = new ParallelOptions
                {
                    MaxDegreeOfParallelism = DomImageParallelism
                };

                try
                {
                    Parallel.ForEach(uniqueImageDicts, parallelOptions, (imgDict) =>
                    {
                        var imgUrl = imgDict["src"]?.ToString() ?? "";
                        if (string.IsNullOrEmpty(imgUrl))
                            return;

                        Interlocked.Increment(ref analyzedCount);

                        try
                        {
                            using var perImageCts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
                            perImageCts.CancelAfter(TimeSpan.FromSeconds(2));

                            using var imgRequest = new HttpRequestMessage(HttpMethod.Get, imgUrl);
                            imgRequest.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
                            using var imgResponse = _httpClient.Send(imgRequest, perImageCts.Token);

                            if (!imgResponse.IsSuccessStatusCode)
                                return;

                            var imgBytes = imgResponse.Content.ReadAsByteArrayAsync(perImageCts.Token).GetAwaiter().GetResult();
                            var imgResult = ClassifyImage(imgBytes);

                            Console.Error.WriteLine($"[INFO] Image classified: {imgUrl} -> {imgResult.Classification}");

                            if (imgResult.Classification == "bad")
                            {
                                badImageBag.Add(new Dictionary<string, object>
                                {
                                    ["url"] = imgUrl,
                                    ["idx"] = imgDict.ContainsKey("idx") ? imgDict["idx"] : -1,
                                    ["confidence"] = imgResult.Confidence,
                                    ["classification"] = imgResult.Classification,
                                    ["element_info"] = new Dictionary<string, object>
                                    {
                                        ["tag"] = imgDict["tag"] ?? "img",
                                        ["id"] = imgDict["id"] ?? "",
                                        ["class"] = imgDict["class"] ?? "",
                                        ["css_selector"] = BuildCssSelector(imgDict["tag"]?.ToString() ?? "img", imgDict["id"]?.ToString() ?? "", imgDict["class"]?.ToString() ?? "")
                                    }
                                });

                                lock (confidenceLock)
                                {
                                    if (imgResult.Confidence > maxBadImageConfidence)
                                    {
                                        maxBadImageConfidence = imgResult.Confidence;
                                    }
                                }
                            }
                        }
                        catch (OperationCanceledException)
                        {
                        }
                        catch
                        {
                        }
                    });
                }
                catch (OperationCanceledException)
                {
                }

                badImageElements.AddRange(badImageBag.Cast<object>());

                imageAnalysis["has_harmful_images"] = badImageElements.Count > 0;
                imageAnalysis["images_analyzed"] = analyzedCount;
                imageAnalysis["bad_images"] = badImageElements.Count;
                imageAnalysis["good_images"] = analyzedCount - badImageElements.Count;
                imageAnalysis["bad_image_elements"] = badImageElements;
            }

            imageAnalysisStopwatch.Stop();
            Console.Error.WriteLine($"[DEBUG] Image analysis duration_ms={imageAnalysisStopwatch.ElapsedMilliseconds}, images_analyzed={imageAnalysis["images_analyzed"]}");

            var finalClassification = "good";
            var finalConfidence = 0.5;
            var finalReasons = new List<string>();

            // TWO-PASS CHECK:
            // 1. First check domain reputation using classifier
            // 2. If domain risk >= 90%, block entire page
            // 3. If domain risk < 90%, analyze and remove offending content
            
            string domainStr = "";
            string domainReputation = "unknown";
            bool inDb = false;
            double domainRiskScore = 0.5;
            bool shouldBlockEntirePage = false;
            
            if (!string.IsNullOrEmpty(url))
            {
                try
                {
                    var uri = new Uri(url);
                    domainStr = uri.Host;
                    
                    // Use the new domain classifier
                    domainRiskScore = ClassifyDomain(domainStr);
                    
                    // Check reputation database for category
                    if (TryGetDomainReputation(domainStr, out domainReputation))
                    {
                        inDb = true;
                    }
                    
                    // If domain risk >= 90%, block entire page immediately
                    if (domainRiskScore >= 0.9)
                    {
                        shouldBlockEntirePage = true;
                        finalClassification = "bad";
                        finalConfidence = domainRiskScore;
                        finalReasons.Add(domainReputation == "phishing" ? "known_phishing_domain" : "domain_high_risk");
                        Console.Error.WriteLine($"[INFO] Domain {domainStr} has high risk ({domainRiskScore:P0}), blocking entire page");
                    }
                }
                catch { }
            }

            // If domain is high risk, skip content analysis and block immediately
            if (!shouldBlockEntirePage)
            {
                // Harmful images always mark page as bad; multiple harmful images escalate to full-page block.
                if (badImageElements.Count > 0)
                {
                    finalClassification = "bad";
                    finalConfidence = Math.Max(finalConfidence, maxBadImageConfidence > 0 ? maxBadImageConfidence : finalConfidence);
                    finalReasons.Add("harmful_images_detected");
                }

                if (badImageElements.Count > 1)
                {
                    shouldBlockEntirePage = true;
                    finalClassification = "bad";
                    finalConfidence = Math.Max(finalConfidence, maxBadImageConfidence > 0 ? maxBadImageConfidence : 0.9);
                    finalReasons.Add("multiple_harmful_images_detected");
                }

                // Harmful text should block the page.
                if (textElementResults.Count > 0)
                {
                    shouldBlockEntirePage = true;
                    finalClassification = "bad";
                    finalConfidence = Math.Max(finalConfidence, highestConfidence > 0 ? highestConfidence : 0.9);
                    finalReasons.Add("harmful_text_detected");
                }

                if (!shouldBlockEntirePage)
                {
                    // For non-blocked pages, confidence should reflect "good" confidence,
                    // which is the inverse of domain risk instead of the default 0.5 fallback.
                    finalConfidence = Math.Clamp(1.0 - domainRiskScore, 0.0, 1.0);
                }
            }

            if (textAnalysis.TryGetValue("classification", out var textClassificationObj)
                && string.Equals(textClassificationObj?.ToString(), "good", StringComparison.OrdinalIgnoreCase)
                && textAnalysis.TryGetValue("confidence", out var textConfidenceObj))
            {
                var textConfidence = 0.5;
                try
                {
                    textConfidence = Convert.ToDouble(textConfidenceObj);
                }
                catch
                {
                }

                // If text pipeline left neutral default confidence, align it with page-level good confidence.
                if (textConfidence <= 0.5)
                {
                    textAnalysis["confidence"] = finalConfidence;
                }
            }

            var response = new Dictionary<string, object>
            {
                ["page_classification"] = finalClassification,
                ["page_confidence"] = finalConfidence,
                ["page_reasons"] = finalReasons.ToArray(),
                ["action_taken"] = shouldBlockEntirePage ? "page_blocked" : "content_removed",
                ["text_analysis"] = textAnalysis,
                ["image_analysis"] = imageAnalysis,
                ["domain_analysis"] = new Dictionary<string, object>
                {
                    ["domain"] = domainStr,
                    ["risk_score"] = domainRiskScore,
                    ["category"] = domainReputation,
                    ["in_database"] = inDb
                },
                ["type"] = "dom",
                ["requested_url"] = url ?? "",
                ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };

            if (requestId.HasValue)
                response["requestId"] = requestId.Value;

            RecordLowReputationVisitIfNeeded(
                url,
                domainStr,
                domainRiskScore,
                "dom_analysis",
                finalClassification,
                finalReasons,
                shouldBlockEntirePage ? "page_blocked" : "content_removed");

            domStopwatch.Stop();
            Console.Error.WriteLine($"[INFO] DOM analysis completed in {domStopwatch.ElapsedMilliseconds} ms for url={url}");

            return SerializeResponse(response);
        }
        catch (Exception ex)
        {
            domStopwatch.Stop();
            Console.Error.WriteLine($"[ERROR] DOM analysis failed after {domStopwatch.ElapsedMilliseconds} ms");
            Console.Error.WriteLine($"[ERROR] DOM analysis error: {ex.Message}");
            return FormatError("dom_analysis_error", ex.Message);
        }
    }

    private static string TruncateForLog(string? value, int maxLength)
    {
        if (string.IsNullOrEmpty(value))
            return string.Empty;

        return value.Length <= maxLength
            ? value
            : value.Substring(0, maxLength) + "...";
    }

    private static string BuildRequestScopeId(double? messageId, string messageType)
    {
        var normalizedType = string.IsNullOrWhiteSpace(messageType) ? "unknown" : messageType;
        if (messageId.HasValue)
        {
            return $"{normalizedType}:{messageId.Value:0.############}";
        }

        return $"{normalizedType}:{Guid.NewGuid().ToString("N").Substring(0, 8)}";
    }

    private static string BuildCssSelector(string tag, string id, string cls)
    {
        var parts = new List<string> { tag };
        
        if (!string.IsNullOrEmpty(id))
            parts.Add($"#{id}");
        if (!string.IsNullOrEmpty(cls))
            parts.Add($".{cls.Replace(" ", ".")}");
        
        return string.Join("", parts);
    }

    private static byte[] HandleClassifyText(JsonElement root)
    {
        if (!root.TryGetProperty("text", out var textElement))
        {
            return FormatError("missing_field", "Missing 'text' field");
        }

        var text = textElement.GetString() ?? "";
        
        if (_textSession == null)
        {
            return FormatError("model_not_loaded", "Text classifier not available");
        }

        var result = ClassifyText(text);
        
        var response = new Dictionary<string, object>
        {
            ["classification"] = result.Classification,
            ["confidence"] = result.Confidence,
            ["reason"] = result.Reason,
            ["type"] = "text",
            ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        };

        var json = JsonSerializer.Serialize(response);
        return Encoding.UTF8.GetBytes(json);
    }

    private static byte[] HandleClassifyImage(JsonElement root)
    {
        if (!root.TryGetProperty("image", out var imageElement))
        {
            return FormatError("missing_field", "Missing 'image' field");
        }

        var imageData = imageElement.GetString() ?? "";
        
        if (_imageSession == null)
        {
            return FormatError("model_not_loaded", "Image classifier not available");
        }

        var result = ClassifyImage(imageData);
        
        var response = new Dictionary<string, object>
        {
            ["classification"] = result.Classification,
            ["confidence"] = result.Confidence,
            ["type"] = "image",
            ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        };

        var json = JsonSerializer.Serialize(response);
        return Encoding.UTF8.GetBytes(json);
    }

    private static byte[] HandleClassifyBoth(JsonElement root)
    {
        string? text = null;
        string? imageBase64 = null;

        if (root.TryGetProperty("text", out var textElement))
            text = textElement.GetString();
        
        if (root.TryGetProperty("image", out var imageElement))
            imageBase64 = imageElement.GetString();

        var result = new Dictionary<string, object>();

        if (!string.IsNullOrEmpty(text) && _textSession != null)
        {
            var textResult = ClassifyText(text);
            result["text"] = new Dictionary<string, object>
            {
                ["classification"] = textResult.Classification,
                ["confidence"] = textResult.Confidence
            };
        }

        if (!string.IsNullOrEmpty(imageBase64) && _imageSession != null)
        {
            var imageResult = ClassifyImage(imageBase64);
            result["image"] = new Dictionary<string, object>
            {
                ["classification"] = imageResult.Classification,
                ["confidence"] = imageResult.Confidence
            };
        }

        var finalClassification = "good";
        if (result.Count > 0)
        {
            if (result.ContainsKey("text"))
            {
                var textResult = (Dictionary<string, object>)result["text"]!;
                if (textResult["classification"]?.ToString() == "bad")
                    finalClassification = "bad";
            }
            if (result.ContainsKey("image"))
            {
                var imageResult = (Dictionary<string, object>)result["image"]!;
                if (imageResult["classification"]?.ToString() == "bad")
                    finalClassification = "bad";
            }
        }

        result["final"] = new Dictionary<string, object>
        {
            ["classification"] = finalClassification,
            ["has_image"] = result.ContainsKey("image"),
            ["has_text"] = result.ContainsKey("text")
        };
        result["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        var json = JsonSerializer.Serialize(result);
        return Encoding.UTF8.GetBytes(json);
    }

    private static byte[] HandleHeartbeat()
    {
        var response = new Dictionary<string, object>
        {
            ["action"] = "pong",
            ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        };

        var json = JsonSerializer.Serialize(response);
        return Encoding.UTF8.GetBytes(json);
    }

    private static byte[] HandleGetStatus()
    {
        var response = new Dictionary<string, object>
        {
            ["status"] = "running",
            ["version"] = "1.0.0",
            ["models"] = new Dictionary<string, string>
            {
                ["image"] = _imageSession != null ? "available" : "unavailable",
                ["text"] = _textSession != null ? "available" : "unavailable"
            },
            ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        };

        var json = JsonSerializer.Serialize(response);
        return Encoding.UTF8.GetBytes(json);
    }

    private static byte[] HandleSaveUserInfo(JsonElement root)
    {
        var email = root.TryGetProperty("email", out var emailElement)
            ? (emailElement.GetString() ?? string.Empty).Trim()
            : string.Empty;
        var phone = root.TryGetProperty("phone", out var phoneElement)
            ? (phoneElement.GetString() ?? string.Empty).Trim()
            : string.Empty;

        if (string.IsNullOrWhiteSpace(email) && string.IsNullOrWhiteSpace(phone))
        {
            return FormatError("invalid_user_info", "At least one contact value is required");
        }

        try
        {
            SaveUserInfo(email, phone);

            var response = new Dictionary<string, object>
            {
                ["action"] = "user_info_saved",
                ["saved"] = true,
                ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };

            return SerializeResponse(response);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to save user info: {ex.Message}");
            return FormatError("save_user_info_failed", ex.Message);
        }
    }

    private static byte[] HandleGetUserInfo()
    {
        try
        {
            var userInfo = GetUserInfo();

            var response = new Dictionary<string, object>
            {
                ["action"] = "user_info",
                ["email"] = userInfo.email,
                ["phone"] = userInfo.phone,
                ["updated_at"] = userInfo.updatedAt,
                ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };

            return SerializeResponse(response);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to get user info: {ex.Message}");
            return FormatError("get_user_info_failed", ex.Message);
        }
    }

    private static byte[] HandleGetLowReputationHistory(JsonElement root)
    {
        try
        {
            var limit = 50;
            if (root.TryGetProperty("limit", out var limitElement))
            {
                try
                {
                    var requestedLimit = limitElement.GetInt32();
                    if (requestedLimit > 0)
                        limit = Math.Min(requestedLimit, 200);
                }
                catch
                {
                }
            }

            var visits = GetLowReputationVisits(limit);

            var response = new Dictionary<string, object>
            {
                ["action"] = "low_reputation_history",
                ["count"] = visits.Count,
                ["items"] = visits,
                ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };

            return SerializeResponse(response);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to get low-reputation history: {ex.Message}");
            return FormatError("get_low_reputation_history_failed", ex.Message);
        }
    }

    private static void EnsureDatabaseInitialized()
    {
        var appDataDir = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        var dbDir = Path.Combine(appDataDir, "SafetyPin");
        Directory.CreateDirectory(dbDir);

        _sqliteDbPath = Path.Combine(dbDir, "safetypin.db");

        lock (_dbLock)
        {
            using var connection = new SqliteConnection($"Data Source={_sqliteDbPath}");
            connection.Open();

            using (var command = connection.CreateCommand())
            {
                command.CommandText = @"
CREATE TABLE IF NOT EXISTS user_info (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  email TEXT,
  phone TEXT,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS low_reputation_visits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  visited_url TEXT NOT NULL,
  domain TEXT NOT NULL,
  risk_score REAL NOT NULL,
  safety_score REAL NOT NULL,
  analysis_type TEXT NOT NULL,
  page_classification TEXT NOT NULL,
    action_taken TEXT,
    blocked_page_shown INTEGER NOT NULL DEFAULT 0,
  reasons TEXT,
  created_at INTEGER NOT NULL
);";
                command.ExecuteNonQuery();
            }

                        try
                        {
                                using var migrationCommand = connection.CreateCommand();
                                migrationCommand.CommandText = "ALTER TABLE low_reputation_visits ADD COLUMN action_taken TEXT;";
                                migrationCommand.ExecuteNonQuery();
                        }
                        catch
                        {
                        }

                        try
                        {
                                using var migrationCommand = connection.CreateCommand();
                                migrationCommand.CommandText = "ALTER TABLE low_reputation_visits ADD COLUMN blocked_page_shown INTEGER NOT NULL DEFAULT 0;";
                                migrationCommand.ExecuteNonQuery();
                        }
                        catch
                        {
                        }
        }

        Console.Error.WriteLine($"[INFO] SQLite initialized at {_sqliteDbPath}");
    }

    private static void SaveUserInfo(string email, string phone)
    {
        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        lock (_dbLock)
        {
            using var connection = new SqliteConnection($"Data Source={_sqliteDbPath}");
            connection.Open();

            using var command = connection.CreateCommand();
            command.CommandText = @"
INSERT INTO user_info (id, email, phone, updated_at)
VALUES (1, $email, $phone, $updated_at)
ON CONFLICT(id) DO UPDATE SET
  email = excluded.email,
  phone = excluded.phone,
  updated_at = excluded.updated_at;";
            command.Parameters.AddWithValue("$email", string.IsNullOrWhiteSpace(email) ? (object)DBNull.Value : email);
            command.Parameters.AddWithValue("$phone", string.IsNullOrWhiteSpace(phone) ? (object)DBNull.Value : phone);
            command.Parameters.AddWithValue("$updated_at", now);
            command.ExecuteNonQuery();
        }
    }

    private static (string email, string phone, long updatedAt) GetUserInfo()
    {
        lock (_dbLock)
        {
            using var connection = new SqliteConnection($"Data Source={_sqliteDbPath}");
            connection.Open();

            using var command = connection.CreateCommand();
            command.CommandText = @"
SELECT email, phone, updated_at
FROM user_info
WHERE id = 1
LIMIT 1;";

            using var reader = command.ExecuteReader();
            if (!reader.Read())
            {
                return (string.Empty, string.Empty, 0);
            }

            var email = reader.IsDBNull(0) ? string.Empty : reader.GetString(0);
            var phone = reader.IsDBNull(1) ? string.Empty : reader.GetString(1);
            var updatedAt = reader.IsDBNull(2) ? 0 : reader.GetInt64(2);
            return (email, phone, updatedAt);
        }
    }

    private static List<Dictionary<string, object>> GetLowReputationVisits(int limit)
    {
        var results = new List<Dictionary<string, object>>();

        lock (_dbLock)
        {
            using var connection = new SqliteConnection($"Data Source={_sqliteDbPath}");
            connection.Open();

            using var command = connection.CreateCommand();
            command.CommandText = @"
SELECT visited_url, domain, risk_score, safety_score, analysis_type, page_classification, action_taken, blocked_page_shown, reasons, created_at
FROM low_reputation_visits
ORDER BY created_at DESC
LIMIT $limit;";
            command.Parameters.AddWithValue("$limit", limit);

            using var reader = command.ExecuteReader();
            while (reader.Read())
            {
                results.Add(new Dictionary<string, object>
                {
                    ["visited_url"] = reader.IsDBNull(0) ? string.Empty : reader.GetString(0),
                    ["domain"] = reader.IsDBNull(1) ? string.Empty : reader.GetString(1),
                    ["risk_score"] = reader.IsDBNull(2) ? 0.0 : reader.GetDouble(2),
                    ["safety_score"] = reader.IsDBNull(3) ? 0.0 : reader.GetDouble(3),
                    ["analysis_type"] = reader.IsDBNull(4) ? string.Empty : reader.GetString(4),
                    ["page_classification"] = reader.IsDBNull(5) ? string.Empty : reader.GetString(5),
                    ["action_taken"] = reader.IsDBNull(6) ? string.Empty : reader.GetString(6),
                    ["blocked_page_shown"] = !reader.IsDBNull(7) && reader.GetInt64(7) == 1,
                    ["reasons"] = reader.IsDBNull(8) ? string.Empty : reader.GetString(8),
                    ["created_at"] = reader.IsDBNull(9) ? 0L : reader.GetInt64(9)
                });
            }
        }

        return results;
    }

    private static void RecordLowReputationVisitIfNeeded(
        string? visitedUrl,
        string? domain,
        double riskScore,
        string analysisType,
        string pageClassification,
        List<string> reasons,
        string actionTaken)
    {
        if (string.IsNullOrWhiteSpace(visitedUrl) || string.IsNullOrWhiteSpace(domain))
            return;

        if (!Uri.TryCreate(visitedUrl, UriKind.Absolute, out var visitedUri))
            return;

        var scheme = visitedUri.Scheme;
        if (!string.Equals(scheme, Uri.UriSchemeHttp, StringComparison.OrdinalIgnoreCase)
            && !string.Equals(scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        var safetyScore = 1.0 - riskScore;
        var blockedPageShown = string.Equals(actionTaken, "page_blocked", StringComparison.OrdinalIgnoreCase);

        if (blockedPageShown || safetyScore < NotificationSafetyThreshold)
        {
            TrySendVisitNotifications(
                visitedUrl,
                domain,
                safetyScore,
                blockedPageShown,
                reasons,
                analysisType,
                pageClassification);
        }

        if (safetyScore >= LowReputationSafetyThreshold && !blockedPageShown)
            return;

        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var reasonsText = reasons.Count == 0 ? string.Empty : string.Join(",", reasons.Distinct());

        try
        {
            lock (_dbLock)
            {
                using var connection = new SqliteConnection($"Data Source={_sqliteDbPath}");
                connection.Open();

                using var command = connection.CreateCommand();
                command.CommandText = @"
INSERT INTO low_reputation_visits
    (visited_url, domain, risk_score, safety_score, analysis_type, page_classification, action_taken, blocked_page_shown, reasons, created_at)
VALUES
    ($visited_url, $domain, $risk_score, $safety_score, $analysis_type, $page_classification, $action_taken, $blocked_page_shown, $reasons, $created_at);";
                command.Parameters.AddWithValue("$visited_url", visitedUrl);
                command.Parameters.AddWithValue("$domain", domain);
                command.Parameters.AddWithValue("$risk_score", riskScore);
                command.Parameters.AddWithValue("$safety_score", safetyScore);
                command.Parameters.AddWithValue("$analysis_type", analysisType);
                command.Parameters.AddWithValue("$page_classification", pageClassification);
                                command.Parameters.AddWithValue("$action_taken", string.IsNullOrWhiteSpace(actionTaken) ? (object)DBNull.Value : actionTaken);
                                command.Parameters.AddWithValue("$blocked_page_shown", blockedPageShown ? 1 : 0);
                command.Parameters.AddWithValue("$reasons", reasonsText);
                command.Parameters.AddWithValue("$created_at", now);
                command.ExecuteNonQuery();
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to record low-reputation visit: {ex.Message}");
        }
    }

    private static void TrySendVisitNotifications(
        string visitedUrl,
        string domain,
        double safetyScore,
        bool blockedPageShown,
        List<string> reasons,
        string analysisType,
        string pageClassification)
    {
        try
        {
            var notificationType = blockedPageShown ? "blocked" : "low_reputation";
            var dedupKey = $"{notificationType}:{domain}".ToLowerInvariant();
            if (!ShouldSendNotificationNow(dedupKey))
            {
                return;
            }

            var userInfo = GetUserInfo();
            if (string.IsNullOrWhiteSpace(userInfo.email) && string.IsNullOrWhiteSpace(userInfo.phone))
            {
                return;
            }

            var safetyPercent = Math.Clamp(safetyScore * 100.0, 0.0, 100.0);
            var reasonsText = reasons.Count == 0 ? "none" : string.Join(", ", reasons.Distinct());
            var eventLabel = blockedPageShown ? "Site blocked" : "Low protection score visit";

            var subject = $"SafeNest Alert: {eventLabel} ({domain})";
            var body =
                $"Event: {eventLabel}\n" +
                $"Domain: {domain}\n" +
                $"URL: {visitedUrl}\n" +
                $"Protection score: {safetyPercent.ToString("0.0", CultureInfo.InvariantCulture)}%\n" +
                $"Analysis: {analysisType}\n" +
                $"Classification: {pageClassification}\n" +
                $"Reasons: {reasonsText}\n" +
                $"Timestamp (UTC): {DateTime.UtcNow:O}";

            var smsBody =
                $"SafeNest alert: {(blockedPageShown ? "site blocked" : "low score site")}. " +
                $"{domain} ({safetyPercent.ToString("0.0", CultureInfo.InvariantCulture)}%).";

            if (!string.IsNullOrWhiteSpace(userInfo.email))
            {
                _ = TrySendEmailNotification(userInfo.email, subject, body);
            }

            if (!string.IsNullOrWhiteSpace(userInfo.phone))
            {
                _ = TrySendSmsNotification(userInfo.phone, smsBody);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to send notifications: {ex.Message}");
        }
    }

    private static bool ShouldSendNotificationNow(string dedupKey)
    {
        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var cooldownSeconds = (long)NotificationCooldown.TotalSeconds;

        if (_lastNotificationByKey.TryGetValue(dedupKey, out var previousSentAt))
        {
            if (now - previousSentAt < cooldownSeconds)
            {
                return false;
            }
        }

        _lastNotificationByKey[dedupKey] = now;
        return true;
    }

    private static bool TrySendEmailNotification(string recipientEmail, string subject, string body)
    {
        try
        {
            var smtpHost = Environment.GetEnvironmentVariable("SAFETYPIN_SMTP_HOST") ?? string.Empty;
            if (string.IsNullOrWhiteSpace(smtpHost))
            {
                Console.Error.WriteLine("[WARNING] Email notification skipped: SAFETYPIN_SMTP_HOST is not set.");
                return false;
            }

            var smtpPortRaw = Environment.GetEnvironmentVariable("SAFETYPIN_SMTP_PORT");
            var smtpPort = 587;
            if (!string.IsNullOrWhiteSpace(smtpPortRaw) && int.TryParse(smtpPortRaw, out var parsedPort) && parsedPort > 0)
            {
                smtpPort = parsedPort;
            }

            var smtpUser = Environment.GetEnvironmentVariable("SAFETYPIN_SMTP_USER") ?? string.Empty;
            var smtpPass = Environment.GetEnvironmentVariable("SAFETYPIN_SMTP_PASS") ?? string.Empty;
            var smtpFrom = Environment.GetEnvironmentVariable("SAFETYPIN_EMAIL_FROM");
            var fromAddress = !string.IsNullOrWhiteSpace(smtpFrom)
                ? smtpFrom
                : (!string.IsNullOrWhiteSpace(smtpUser) ? smtpUser : "alerts@safenest.local");

            var enableSslRaw = Environment.GetEnvironmentVariable("SAFETYPIN_SMTP_SSL") ?? "true";
            var enableSsl = !string.Equals(enableSslRaw, "false", StringComparison.OrdinalIgnoreCase);

            using var smtpClient = new SmtpClient(smtpHost, smtpPort)
            {
                EnableSsl = enableSsl
            };

            if (!string.IsNullOrWhiteSpace(smtpUser))
            {
                smtpClient.Credentials = new NetworkCredential(smtpUser, smtpPass);
            }

            using var message = new MailMessage(fromAddress, recipientEmail, subject, body);
            smtpClient.Send(message);
            Console.Error.WriteLine($"[INFO] Email notification sent to {recipientEmail}");
            return true;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Email notification failed: {ex.Message}");
            return false;
        }
    }

    private static bool TrySendSmsNotification(string recipientPhone, string message)
    {
        try
        {
            var accountSid = Environment.GetEnvironmentVariable("SAFETYPIN_TWILIO_ACCOUNT_SID") ?? string.Empty;
            var authToken = Environment.GetEnvironmentVariable("SAFETYPIN_TWILIO_AUTH_TOKEN") ?? string.Empty;
            var fromPhone = Environment.GetEnvironmentVariable("SAFETYPIN_TWILIO_FROM_PHONE") ?? string.Empty;

            if (string.IsNullOrWhiteSpace(accountSid) || string.IsNullOrWhiteSpace(authToken) || string.IsNullOrWhiteSpace(fromPhone))
            {
                Console.Error.WriteLine("[WARNING] SMS notification skipped: Twilio environment variables are not fully configured.");
                return false;
            }

            using var request = new HttpRequestMessage(
                HttpMethod.Post,
                $"https://api.twilio.com/2010-04-01/Accounts/{accountSid}/Messages.json");

            var basicToken = Convert.ToBase64String(Encoding.UTF8.GetBytes($"{accountSid}:{authToken}"));
            request.Headers.Authorization = new AuthenticationHeaderValue("Basic", basicToken);
            request.Content = new FormUrlEncodedContent(new Dictionary<string, string>
            {
                ["To"] = recipientPhone,
                ["From"] = fromPhone,
                ["Body"] = message
            });

            using var response = _httpClient.Send(request);
            if (!response.IsSuccessStatusCode)
            {
                var errorBody = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
                Console.Error.WriteLine($"[ERROR] SMS notification failed: {response.StatusCode} {errorBody}");
                return false;
            }

            Console.Error.WriteLine($"[INFO] SMS notification sent to {recipientPhone}");
            return true;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] SMS notification failed: {ex.Message}");
            return false;
        }
    }

    private static byte[] FormatError(string errorType, string message, double? messageId = null)
    {
        var response = new Dictionary<string, object>
        {
            ["error"] = errorType,
            ["message"] = message,
            ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        };
        
        if (messageId.HasValue)
        {
            response["messageId"] = messageId.Value;
        }

        var json = JsonSerializer.Serialize(response);
        return Encoding.UTF8.GetBytes(json);
    }

    private static byte[] SerializeResponse(Dictionary<string, object> response)
    {
        if (_currentMessageId.HasValue)
        {
            response["messageId"] = _currentMessageId.Value;
        }
        var json = JsonSerializer.Serialize(response);
        return Encoding.UTF8.GetBytes(json);
    }

    private static ClassificationResult ClassifyText(string text)
    {
        try
        {
            var inputs = TokenizeText(text);
            
            var inputIdsArray = inputs.inputIds;
            var attentionMaskArray = inputs.attentionMask;

            var inputTensorIds = new DenseTensor<long>(new[] { 1, inputIdsArray.Length });
            var inputAttentionMask = new DenseTensor<float>(new[] { 1, attentionMaskArray.Length });
            
            for (int i = 0; i < inputIdsArray.Length; i++)
            {
                inputTensorIds[0, i] = inputIdsArray[i];
                inputAttentionMask[0, i] = attentionMaskArray[i];
            }

            var inputTensors = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensorIds),
                NamedOnnxValue.CreateFromTensor("attention_mask", inputAttentionMask)
            };

            using var outputs = _textSession!.Run(inputTensors);
            
            var classLogits = outputs[0].AsEnumerable<float>().ToArray();
            var reasonLogits = outputs[1].AsEnumerable<float>().ToArray();

            var classProbs = Softmax(classLogits);
            var reasonProbs = Softmax(reasonLogits);

            var classPred = Array.IndexOf(classProbs, classProbs.Max());
            var reasonPred = Array.IndexOf(reasonProbs, reasonProbs.Max());

            var classification = classPred == 0 ? "bad" : "good";
            var confidence = classProbs[classPred];
            var reason = _reasonNames.GetValueOrDefault(reasonPred, "unknown");

            return new ClassificationResult
            {
                Classification = classification,
                Confidence = confidence,
                Reason = reason
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Text classification error: {ex.Message}");
            return new ClassificationResult
            {
                Classification = "good",
                Confidence = 0.5,
                Reason = "classification_error"
            };
        }
    }

    private static double ClassifyDomain(string domain)
    {
        domain = NormalizeDomain(domain);
        if (string.IsNullOrEmpty(domain))
            return 0.5;

        // First check reputation database
        if (_domainReputation.Count > 0)
        {
            if (TryGetDomainReputation(domain, out var reputation))
            {
                // If known phishing, return high risk (0.9)
                // If known legitimate, return very low risk (0.01 = 99% safety)
                return reputation == "phishing" ? 0.9 : 0.01;
            }
        }

        // Use ONNX model if available
        if (_domainSession != null && _charToIdx.Count > 0)
        {
            try
            {
                // Encode domain
                var inputIds = new long[_domainMaxLength];
                for (int i = 0; i < _domainMaxLength; i++)
                {
                    if (i < domain.Length && _charToIdx.TryGetValue(domain[i], out var mappedIndex))
                    {
                        inputIds[i] = mappedIndex;
                    }
                    else
                    {
                        inputIds[i] = 1; // <UNK>
                    }
                }

                var inputTensor = new DenseTensor<long>(inputIds, new[] { 1, _domainMaxLength });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
                };

                using var outputs = _domainSession.Run(inputs);
                var logits = outputs.FirstOrDefault()?.AsEnumerable<float>().ToArray();
                
                if (logits != null && logits.Length >= 2)
                {
                    // Apply softmax
                    var exp = logits.Select(x => (float)Math.Exp(x)).ToArray();
                    var sum = exp.Sum();
                    var probs = exp.Select(x => x / sum).ToArray();
                    
                    // Return probability of being phishing (index 0)
                    return probs[0];
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Domain classification error: {ex.Message}");
            }
        }

        // Default: unknown domain, use neutral risk
        return 0.5;
    }

    private static string NormalizeDomain(string domain)
    {
        if (string.IsNullOrWhiteSpace(domain))
            return string.Empty;

        return domain.Trim().TrimEnd('.').ToLowerInvariant();
    }

    private static IEnumerable<string> GetDomainCandidates(string domain)
    {
        var normalized = NormalizeDomain(domain);
        if (string.IsNullOrEmpty(normalized))
            yield break;

        yield return normalized;

        if (normalized.StartsWith("www."))
        {
            var withoutWww = normalized.Substring(4);
            if (!string.IsNullOrEmpty(withoutWww))
                yield return withoutWww;
        }
        else
        {
            yield return $"www.{normalized}";
        }
    }

    private static bool TryGetDomainReputation(string domain, out string reputation)
    {
        reputation = "unknown";

        if (_domainReputation.Count == 0)
            return false;

        foreach (var candidate in GetDomainCandidates(domain))
        {
            if (_domainReputation.TryGetValue(candidate, out var matchedReputation))
            {
                reputation = matchedReputation;
                return true;
            }
        }

        return false;
    }

    private static (long[] inputIds, int[] attentionMask) TokenizeText(string text)
    {
        var tokens = new List<long>();
        var attention = new List<int>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return CreateEmptyTokens();
        }

        var cleanedText = CleanText(text);
        var words = SplitIntoWords(cleanedText);

        foreach (var word in words)
        {
            if (tokens.Count >= MaxTextLength - 2) break;

            var subTokens = TokenizeWordPiece(word);
            
            foreach (var subToken in subTokens)
            {
                if (tokens.Count >= MaxTextLength - 2) break;
                tokens.Add(VocabToId(subToken));
                attention.Add(1);
            }
        }

        tokens.Insert(0, VocabCls);
        tokens.Add(VocabSep);
        attention.Insert(0, 1);
        attention.Add(1);

        while (tokens.Count < MaxTextLength)
        {
            tokens.Add(VocabPad);
            attention.Add(0);
        }

        return (tokens.ToArray(), attention.ToArray());
    }

    private static (long[] inputIds, int[] attentionMask) CreateEmptyTokens()
    {
        var tokens = new long[MaxTextLength];
        var attention = new int[MaxTextLength];
        
        tokens[0] = VocabCls;
        tokens[1] = VocabSep;
        attention[0] = 1;
        attention[1] = 1;
        
        return (tokens, attention);
    }

    private static string CleanText(string text)
    {
        text = text.ToLower();
        text = Regex.Replace(text, @"\s+", " ");
        text = text.Trim();
        return text;
    }

    private static string[] SplitIntoWords(string text)
    {
        var words = new List<string>();
        var currentWord = new StringBuilder();
        
        foreach (var c in text)
        {
            if (char.IsLetterOrDigit(c))
            {
                currentWord.Append(c);
            }
            else if (currentWord.Length > 0)
            {
                words.Add(currentWord.ToString());
                currentWord.Clear();
            }
            
            if (currentWord.Length > 0 && (c == '\'' || c == '-' || c == '.'))
            {
            }
            else if (!char.IsLetterOrDigit(c) && currentWord.Length > 0)
            {
                words.Add(currentWord.ToString());
                currentWord.Clear();
            }
        }
        
        if (currentWord.Length > 0)
        {
            words.Add(currentWord.ToString());
        }
        
        return words.ToArray();
    }

    private static List<string> TokenizeWordPiece(string word)
    {
        var result = new List<string>();
        
        if (string.IsNullOrEmpty(word))
        {
            return result;
        }

        if (_vocabulary.Count > 0 && _vocabulary.ContainsKey(word))
        {
            result.Add(word);
            return result;
        }

        var start = 0;
        var substrings = new List<string>();
        
        while (start < word.Length)
        {
            var end = word.Length;
            var hasMatch = false;
            
            while (start < end)
            {
                var substr = start == 0 ? word.Substring(start, end - start) : "##" + word.Substring(start, end - start);
                
                if (start > 0)
                {
                    substr = "##" + word.Substring(start, end - start);
                }
                
                if (_vocabulary.Count > 0 && _vocabulary.ContainsKey(substr))
                {
                    substrings.Add(substr);
                    hasMatch = true;
                    break;
                }
                
                end--;
            }
            
            if (!hasMatch)
            {
                substrings.Add("[UNK]");
                break;
            }
            
            start = end;
        }
        
        if (substrings.Count == 0)
        {
            substrings.Add("[UNK]");
        }
        
        return substrings;
    }

    private static long VocabToId(string token)
    {
        if (_vocabulary.TryGetValue(token, out var id))
            return id;
        return VocabUnknown;
    }

    private static ClassificationResult ClassifyImage(string imageBase64)
    {
        try
        {
            if (imageBase64.StartsWith("data:image"))
            {
                var commaIndex = imageBase64.IndexOf(',');
                if (commaIndex >= 0)
                    imageBase64 = imageBase64.Substring(commaIndex + 1);
            }

            var imageBytes = Convert.FromBase64String(imageBase64);
            return ClassifyImage(imageBytes);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Image classification error: {ex.Message}");
            return new ClassificationResult
            {
                Classification = "good",
                Confidence = 0.5,
                Reason = "classification_error"
            };
        }
    }

    private static ClassificationResult ClassifyImage(byte[] imageBytes)
    {
        try
        {

            using var image = Image.Load<Rgb24>(imageBytes);
            
            image.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(128, 128),
                Mode = ResizeMode.Crop
            }));

            var tensor = new float[1 * 3 * 128 * 128];
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };

            for (int y = 0; y < 128; y++)
            {
                for (int x = 0; x < 128; x++)
                {
                    var pixel = image[x, y];
                    tensor[0 * 128 * 128 + y * 128 + x] = (pixel.R / 255f - mean[0]) / std[0];
                    tensor[1 * 128 * 128 + y * 128 + x] = (pixel.G / 255f - mean[1]) / std[1];
                    tensor[2 * 128 * 128 + y * 128 + x] = (pixel.B / 255f - mean[2]) / std[2];
                }
            }

            var inputTensor = new DenseTensor<float>(new[] { 1, 3, 128, 128 });
            int idx = 0;
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < 128; y++)
                {
                    for (int x = 0; x < 128; x++)
                    {
                        inputTensor[0, c, y, x] = tensor[idx++];
                    }
                }
            }
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var outputs = _imageSession!.Run(inputs);
            var logits = outputs[0].AsEnumerable<float>().ToArray();
            var probs = Softmax(logits);

            var predicted = Array.IndexOf(probs, probs.Max());
            var classification = predicted == 0 ? "bad" : "good";
            var confidence = probs[predicted];

            return new ClassificationResult
            {
                Classification = classification,
                Confidence = confidence,
                Reason = ""
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Image classification error: {ex.Message}");
            return new ClassificationResult
            {
                Classification = "good",
                Confidence = 0.5,
                Reason = "classification_error"
            };
        }
    }

    private static float[] Softmax(float[] values)
    {
        var max = values.Max();
        var exp = values.Select(x => (float)Math.Exp(x - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(x => x / sum).ToArray();
    }

    private static string GetXPath(HtmlNode node)
    {
        if (node == null)
            return "";

        var path = new List<string>();
        var current = node;

        while (current != null)
        {
            var name = current.Name;
            var parent = current.ParentNode;

            if (parent != null)
            {
                var siblings = parent.ChildNodes.Where(n => n.Name == name).ToList();
                if (siblings.Count > 1)
                {
                    var index = siblings.IndexOf(current) + 1;
                    path.Insert(0, $"{name}[{index}]");
                }
                else
                {
                    path.Insert(0, name);
                }
            }
            else
            {
                path.Insert(0, name);
            }

            if (name == "html" || name == "body")
                break;

            current = parent;
        }

        return "/" + string.Join("/", path);
    }

    private static Dictionary<string, object> GetElementInfo(HtmlNode node)
    {
        var info = new Dictionary<string, object>
        {
            ["tag"] = node.Name,
            ["id"] = node.GetAttributeValue("id", ""),
            ["class"] = node.GetAttributeValue("class", ""),
            ["xpath"] = GetXPath(node)
        };

        var dataAttributes = new Dictionary<string, string>();
        foreach (var attr in node.Attributes)
        {
            if (attr.Name.StartsWith("data-"))
            {
                dataAttributes[attr.Name] = attr.Value;
            }
        }
        info["data_attributes"] = dataAttributes;

        var selectorParts = new List<string> { node.Name };
        var id = node.GetAttributeValue("id", "");
        var cls = node.GetAttributeValue("class", "");
        
        if (!string.IsNullOrEmpty(id))
            selectorParts.Add($"#{id}");
        if (!string.IsNullOrEmpty(cls))
            selectorParts.Add($".{cls.Replace(" ", ".")}");
        
        info["css_selector"] = string.Join("", selectorParts);

        return info;
    }

    private static void InstallNativeHost(string[] args)
    {
        var extensionId = args.Length > 0 ? args[0] : "nggcdgkdicaadoeicjeehijpfbdmopeg";
        var binaryPath = args.Length > 1 && !string.IsNullOrWhiteSpace(args[1])
            ? args[1]
            : (Environment.ProcessPath ?? System.Reflection.Assembly.GetExecutingAssembly().Location);

        if (!string.IsNullOrWhiteSpace(binaryPath))
        {
            binaryPath = Path.GetFullPath(binaryPath);
        }

        Console.Error.WriteLine($"[INFO] Installing native host for extension: {extensionId}");
        Console.Error.WriteLine($"[INFO] Binary path: {binaryPath}");

        if (string.IsNullOrWhiteSpace(binaryPath) || !File.Exists(binaryPath))
        {
            Console.Error.WriteLine("[WARNING] Binary path does not exist; manifest path entry may be invalid");
        }

        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (string.IsNullOrWhiteSpace(home))
        {
            home = Environment.GetEnvironmentVariable("HOME") ?? string.Empty;
        }
        
        string manifestDir;
        string manifestPath;

        if (OperatingSystem.IsWindows())
        {
            manifestDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Google", "Chrome", "NativeMessagingHosts");
            manifestPath = Path.Combine(manifestDir, "com.safetypin.native.json");
        }
        else if (OperatingSystem.IsMacOS())
        {
            manifestDir = Path.Combine(home, "Library", "Application Support", "Google", "Chrome", "NativeMessagingHosts");
            manifestPath = Path.Combine(manifestDir, "com.safetypin.native.json");
        }
        else
        {
            manifestDir = Path.Combine(home, ".config", "google-chrome", "NativeMessagingHosts");
            manifestPath = Path.Combine(manifestDir, "com.safetypin.native.json");
        }

        Directory.CreateDirectory(manifestDir);

        var manifest = new Dictionary<string, object>
        {
            ["name"] = "com.safetypin.native",
            ["description"] = "Child Safety Monitor Native Host",
            ["path"] = binaryPath,
            ["type"] = "stdio",
            ["allowed_origins"] = new[] { $"chrome-extension://{extensionId}/" }
        };

        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(manifestPath, json);

        Console.Error.WriteLine($"[INFO] Manifest written to: {manifestPath}");
    }

    private class ClassificationResult
    {
        public string Classification { get; set; } = "good";
        public double Confidence { get; set; } = 0.5;
        public string Reason { get; set; } = "";
    }
}

public enum AppLogLevel
{
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    None = 5
}

public static class AppLogger
{
    private static readonly object Sync = new();
    private static readonly AsyncLocal<string?> CurrentRequestScope = new();
    private static StreamWriter? _writer;
    private static string _logFilePath = string.Empty;
    private static long _maxFileBytes;
    private static int _maxRotatedFiles;
    private static AppLogLevel _minimumLevel;
    private static TextWriter? _originalErrorWriter;
    private static bool _initialized;

    public static void Initialize()
    {
        lock (Sync)
        {
            if (_initialized)
                return;

            _minimumLevel = ParseLogLevel(Environment.GetEnvironmentVariable("SAFETYPIN_LOG_LEVEL")) ?? AppLogLevel.Trace;
            _maxFileBytes = ParseLong(Environment.GetEnvironmentVariable("SAFETYPIN_LOG_MAX_BYTES"), 10 * 1024 * 1024);
            _maxRotatedFiles = (int)ParseLong(Environment.GetEnvironmentVariable("SAFETYPIN_LOG_MAX_FILES"), 5);

            _logFilePath = Environment.GetEnvironmentVariable("SAFETYPIN_LOG_PATH") ?? GetDefaultLogPath();
            var logDir = Path.GetDirectoryName(_logFilePath) ?? AppDomain.CurrentDomain.BaseDirectory;
            Directory.CreateDirectory(logDir);

            _writer = CreateWriter(_logFilePath);

            _originalErrorWriter = Console.Error;
            Console.SetError(new InterceptingErrorWriter(_originalErrorWriter));
            _initialized = true;

            Write(AppLogLevel.Info, $"Logger initialized: path={_logFilePath}, max_bytes={_maxFileBytes}, max_files={_maxRotatedFiles}, min_level={_minimumLevel}");
        }
    }

    public static void Shutdown()
    {
        lock (Sync)
        {
            if (!_initialized)
                return;

            try
            {
                Write(AppLogLevel.Info, "Logger shutdown requested");
                _writer?.Flush();
                _writer?.Dispose();
            }
            catch
            {
            }
            finally
            {
                _writer = null;
                if (_originalErrorWriter != null)
                {
                    Console.SetError(_originalErrorWriter);
                }
                _initialized = false;
            }
        }
    }

    public static void Write(AppLogLevel level, string message)
    {
        lock (Sync)
        {
            if (!_initialized || level < _minimumLevel)
                return;

            if (_writer == null)
                _writer = CreateWriter(_logFilePath);

            RotateIfNeeded(message);
            var timestamp = DateTimeOffset.Now.ToString("yyyy-MM-dd HH:mm:ss.fff zzz");
            var scopeSuffix = string.IsNullOrWhiteSpace(CurrentRequestScope.Value) ? string.Empty : $" [req:{CurrentRequestScope.Value}]";
            var line = $"{timestamp} [{level.ToString().ToUpperInvariant()}]{scopeSuffix} {message}";
            _writer!.WriteLine(line);
            _writer.Flush();
        }
    }

    public static IDisposable BeginRequestScope(string scopeId)
    {
        var previousScope = CurrentRequestScope.Value;
        CurrentRequestScope.Value = scopeId;
        return new RequestScope(previousScope);
    }

    public static void SetRequestScope(string scopeId)
    {
        CurrentRequestScope.Value = scopeId;
    }

    public static void ClearRequestScope()
    {
        CurrentRequestScope.Value = null;
    }

    public static void WriteRawConsoleError(string message)
    {
        var mappedLevel = InferLevelFromMessage(message);
        Write(mappedLevel, StripLevelPrefix(message));
    }

    private static void RotateIfNeeded(string nextMessage)
    {
        if (_writer == null)
            return;

        try
        {
            var estimatedBytes = Encoding.UTF8.GetByteCount(nextMessage) + 128;
            var currentLength = _writer.BaseStream.Length;

            if (currentLength + estimatedBytes < _maxFileBytes)
                return;

            _writer.Flush();
            _writer.Dispose();
            _writer = null;

            var oldest = _logFilePath + "." + _maxRotatedFiles;
            if (File.Exists(oldest))
                File.Delete(oldest);

            for (int index = _maxRotatedFiles - 1; index >= 1; index--)
            {
                var source = _logFilePath + "." + index;
                var target = _logFilePath + "." + (index + 1);
                if (File.Exists(source))
                {
                    File.Move(source, target, true);
                }
            }

            if (File.Exists(_logFilePath))
            {
                File.Move(_logFilePath, _logFilePath + ".1", true);
            }

            _writer = CreateWriter(_logFilePath);
            var rotationNote = $"{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff zzz} [INFO] Log rotated";
            _writer.WriteLine(rotationNote);
            _writer.Flush();
        }
        catch
        {
        }
    }

    private static StreamWriter CreateWriter(string path)
    {
        var fileStream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
        return new StreamWriter(fileStream, Encoding.UTF8) { AutoFlush = true };
    }

    private static string GetDefaultLogPath()
    {
        string logDir;

        if (OperatingSystem.IsWindows())
        {
            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            logDir = Path.Combine(localAppData, "SafetyPin", "Logs");
        }
        else if (OperatingSystem.IsMacOS())
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            logDir = Path.Combine(home, "Library", "Logs", "SafetyPin");
        }
        else
        {
            var home = Environment.GetEnvironmentVariable("HOME") ?? "/tmp";
            logDir = Path.Combine(home, ".local", "share", "safetypin", "logs");
        }

        return Path.Combine(logDir, "native-host.log");
    }

    private static AppLogLevel InferLevelFromMessage(string message)
    {
        if (message.Contains("[TRACE]", StringComparison.OrdinalIgnoreCase)) return AppLogLevel.Trace;
        if (message.Contains("[DEBUG]", StringComparison.OrdinalIgnoreCase)) return AppLogLevel.Debug;
        if (message.Contains("[INFO]", StringComparison.OrdinalIgnoreCase)) return AppLogLevel.Info;
        if (message.Contains("[WARNING]", StringComparison.OrdinalIgnoreCase) || message.Contains("[WARN]", StringComparison.OrdinalIgnoreCase)) return AppLogLevel.Warning;
        if (message.Contains("[ERROR]", StringComparison.OrdinalIgnoreCase)) return AppLogLevel.Error;
        return AppLogLevel.Info;
    }

    private static string StripLevelPrefix(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
            return string.Empty;

        var trimmed = message.TrimStart();
        var prefixes = new[] { "[TRACE]", "[DEBUG]", "[INFO]", "[WARNING]", "[WARN]", "[ERROR]" };
        foreach (var prefix in prefixes)
        {
            if (trimmed.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return trimmed.Substring(prefix.Length).TrimStart();
            }
        }

        return message;
    }

    private static AppLogLevel? ParseLogLevel(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return value.Trim().ToUpperInvariant() switch
        {
            "TRACE" => AppLogLevel.Trace,
            "DEBUG" => AppLogLevel.Debug,
            "INFO" => AppLogLevel.Info,
            "WARNING" => AppLogLevel.Warning,
            "WARN" => AppLogLevel.Warning,
            "ERROR" => AppLogLevel.Error,
            "NONE" => AppLogLevel.None,
            _ => null
        };
    }

    private static long ParseLong(string? value, long fallback)
    {
        return long.TryParse(value, out var parsed) && parsed > 0 ? parsed : fallback;
    }

    private sealed class InterceptingErrorWriter : TextWriter
    {
        private readonly TextWriter _inner;

        public InterceptingErrorWriter(TextWriter inner)
        {
            _inner = inner;
        }

        public override Encoding Encoding => Encoding.UTF8;

        public override void WriteLine(string? value)
        {
            var message = value ?? string.Empty;
            try
            {
                AppLogger.WriteRawConsoleError(message);
            }
            catch
            {
            }

            _inner.WriteLine(message);
        }

        public override void Write(string? value)
        {
            _inner.Write(value);
        }
    }

    private sealed class RequestScope : IDisposable
    {
        private readonly string? _previousScope;
        private bool _disposed;

        public RequestScope(string? previousScope)
        {
            _previousScope = previousScope;
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            CurrentRequestScope.Value = _previousScope;
            _disposed = true;
        }
    }
}
