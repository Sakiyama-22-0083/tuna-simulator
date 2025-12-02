using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// 遺伝的アルゴリズムを用いて Boids パラメータを最適化するマネージャ。
/// 各個体の遺伝子セットを順番に適用し、10 秒動画の server スコアを適応度として利用する。
/// </summary>
public class GeneticAlgorithmManager : MonoBehaviour
{
    #region GA 設定
    [Header("遺伝的アルゴリズム設定")]
    [Tooltip("個体数（1世代あたりの評価回数）")]
    public int populationSize = 20;

    [Tooltip("エリート選択数")]
    public int eliteCount = 2;

    [Tooltip("突然変異率 [0, 1]")]
    [Range(0f, 1f)]
    public float mutationRate = 0.1f;

    [Tooltip("突然変異の強さ [0, 1]")]
    [Range(0f, 1f)]
    public float mutationStrength = 0.2f;

    [Tooltip("交叉率 [0, 1]")]
    [Range(0f, 1f)]
    public float crossoverRate = 0.8f;

    [Tooltip("最大世代数（0で無限）")]
    public int maxGenerations = 100;

    [Tooltip("自動実行（1世代終了後に次世代を自動評価）")]
    public bool autoRun = true;
    #endregion

    private void LogStatus(string message)
    {
        Debug.Log($"[GA] {message}");
    }

    #region シミュレーション設定
    [Header("シミュレーション設定")]
    [Tooltip("GeneticBoidAgent のプレハブ（評価に用いる Boids 群れ）")]
    public GameObject agentPrefab;

    [Tooltip("初期配置時の半径（Transform 原点を中心としたランダム分布）")]
    public float spawnRange = 10f;

    [Tooltip("初期配置時の最大高さ（ローカル Y 範囲）")]
    public float spawnHeight = 5f;

    [Tooltip("各個体の評価に同時使用するエージェント数")]
    [Min(1)] public int agentsPerEvaluation = 100;

    [Tooltip("スポーン位置を決める際の最大試行回数")]
    public int spawnMaxAttempts = 64;

    [Tooltip("スポーン時の衝突判定に使用するレイヤーマスク（0 で判定しない）")]
    public LayerMask spawnCollisionMask = Physics.DefaultRaycastLayers;

    [Tooltip("Collider を取得できない場合に使用する半径（メートル）")]
    public float spawnFallbackRadius = 1.5f;
    #endregion

    #region 外部連携
    [Header("外部連携")]
    [SerializeField, Tooltip("動画録画・アップロードを担当する VideoRecorder。未指定の場合はシーンから検索する。")]
    private VideoRecorder videoRecorder;

    [Header("評価待機設定")]
    [SerializeField, Tooltip("サーバーのスコア応答を待機する最大秒数。0以下で無制限に待つ。")]
    private float serverScoreTimeoutSeconds = 180f;

    [Header("サーバー接続設定")]
    [SerializeField, Tooltip("サーバーのヘルスチェック先 URL。HTTP 200 応答で接続完了とみなします。")]
    private string serverHealthUrl = "http://127.0.0.1:8000/health";

    [SerializeField, Tooltip("ヘルスチェック再試行間隔（秒）。")]
    private float serverHealthRetryIntervalSeconds = 2f;

    [SerializeField, Tooltip("ヘルスチェックリクエストのタイムアウト秒数。")]
    private int serverHealthRequestTimeoutSeconds = 5;
    #endregion

    #region 内部状態
    private readonly List<GeneticBoidAgent> simulationAgents = new();
    public event System.Action<int, int> OnIndividualEvaluationStarted;
    public IReadOnlyList<GeneticBoidAgent> ActiveAgents => simulationAgents;

    private class Individual
    {
        public float[] Genes;
        public float Fitness;
    }

    private readonly List<Individual> individuals = new();

    private int currentGeneration = 0;
    private int currentIndividualIndex = -1;
    private bool generationInProgress = false;
    private bool waitingForServerScore = false;
    private Coroutine serverScoreTimeoutCoroutine;
    private bool serverReady = false;
    private bool pendingEvaluationStart = false;
    private Coroutine serverHealthCoroutine;
    private bool hasLoggedServerWait = false;

    private float bestFitness = 0f;
    private float avgFitness = 0f;
    private float[] bestGenes = new float[GeneticBoidAgent.GeneCount];

    private Bounds agentPrefabBounds;
    private bool agentBoundsInitialized = false;
    private string csvFilePath;
    private bool csvHeaderWritten = false;
    private string populationCsvFilePath;
    private bool populationCsvHeaderWritten = false;
    #endregion

    #region ログ設定
    [Header("ログ設定")]
    [Tooltip("詳細ログを表示")]
    public bool enableDetailedLogging = true;

    [Tooltip("世代ごとにベスト遺伝子を表示")]
    public bool logBestGenes = true;
    #endregion

    #region CSV 出力
    [Header("CSV ログ設定")]
    [SerializeField, Tooltip("世代ごとのベスト遺伝子と適応度を CSV に書き出すかどうか。")]
    private bool enableCsvLogging = false;

    [SerializeField, Tooltip("CSV ファイル名。プロジェクトルート直下の Logs フォルダに保存されます。")]
    private string csvFileName = "ga_progress.csv";

    [SerializeField, Tooltip("シーン開始時に既存の CSV を上書きするかどうか。無効の場合は追記します。")]
    private bool overwriteCsvOnStart = true;

    [SerializeField, Tooltip("世代ごとに生成された全個体の遺伝子を記録する CSV ファイル名。Logs フォルダに保存されます。")]
    private string populationCsvFileName = "ga_population.csv";
    #endregion

    #region Unity ライフサイクル
    private void Awake()
    {
        if (videoRecorder == null)
        {
            videoRecorder = FindObjectOfType<VideoRecorder>();
        }
    }

    private void Start()
    {
        if (agentPrefab == null)
        {
            Debug.LogError("[GA] Agent prefab not assigned!");
            return;
        }

        CacheAgentPrefabBounds();
        InitializeSimulationAgents();
        InitializePopulationGenes();
        InitializeCsvLogging();
        WritePopulationParametersToCsv(currentGeneration, individuals);

        if (autoRun)
        {
            QueueEvaluationStart();
        }
    }

    private void OnDisable()
    {
        CancelServerScoreTimeoutWatch();

        if (serverHealthCoroutine != null)
        {
            StopCoroutine(serverHealthCoroutine);
            serverHealthCoroutine = null;
        }
    }
    #endregion

    #region 初期化
    private void InitializeSimulationAgents()
    {
        CacheAgentPrefabBounds();

        foreach (var agent in simulationAgents)
        {
            if (agent != null)
            {
                Destroy(agent.gameObject);
            }
        }
        simulationAgents.Clear();

        int desiredCount = Mathf.Max(1, agentsPerEvaluation);

        for (int i = 0; i < desiredCount; i++)
        {
            Quaternion spawnRotation;
            Vector3 spawnPosition;

            if (!TryGenerateSpawnPose(out spawnPosition, out spawnRotation))
            {
                Vector3 fallbackLocal = GenerateRandomSpawnLocalPosition();
                spawnPosition = transform.TransformPoint(fallbackLocal);
                spawnRotation = transform.rotation * Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

                if (enableDetailedLogging)
                {
                    Debug.LogWarning($"[GA] Failed to find collision-free spawn for agent {i}. Using fallback position.");
                }
            }

            GameObject agentObj = Instantiate(agentPrefab, spawnPosition, spawnRotation, transform);
            agentObj.name = $"BoidAgent_Active_{i:00}";

            if (agentObj.TryGetComponent(out GeneticBoidAgent agent))
            {
                agent.ResetFitness();
                simulationAgents.Add(agent);
            }
            else
            {
                Debug.LogError($"[GA] Agent prefab missing GeneticBoidAgent component ({agentObj.name})");
            }
        }
    }

    private void InitializePopulationGenes()
    {
        individuals.Clear();
        for (int i = 0; i < populationSize; i++)
        {
            individuals.Add(new Individual
            {
                Genes = GeneticBoidAgent.GenerateRandomGenes(),
                Fitness = 0f
            });
        }
    }
    #endregion

    #region スポーン補助
    private void EnsureSimulationAgentsReady()
    {
        int desiredCount = Mathf.Max(1, agentsPerEvaluation);
        if (simulationAgents.Count != desiredCount || simulationAgents.Any(agent => agent == null))
        {
            InitializeSimulationAgents();
        }
    }

    private void CacheAgentPrefabBounds()
    {
        if (agentBoundsInitialized || agentPrefab == null)
        {
            return;
        }

        var boxColliders = agentPrefab.GetComponentsInChildren<BoxCollider>(true);
        Bounds calculatedBounds = new Bounds(Vector3.zero, Vector3.zero);
        bool hasBounds = false;

        if (boxColliders != null && boxColliders.Length > 0)
        {
            foreach (BoxCollider box in boxColliders)
            {
                if (box == null)
                {
                    continue;
                }

                Transform colTransform = box.transform;
                Vector3 localCenter = box.center;
                Vector3 halfSize = box.size * 0.5f;

                for (int x = -1; x <= 1; x += 2)
                {
                    for (int y = -1; y <= 1; y += 2)
                    {
                        for (int z = -1; z <= 1; z += 2)
                        {
                            Vector3 localCorner = localCenter + Vector3.Scale(halfSize, new Vector3(x, y, z));
                            Vector3 rootSpace = colTransform.TransformPoint(localCorner);

                            if (!hasBounds)
                            {
                                calculatedBounds = new Bounds(rootSpace, Vector3.zero);
                                hasBounds = true;
                            }
                            else
                            {
                                calculatedBounds.Encapsulate(rootSpace);
                            }
                        }
                    }
                }
            }
        }

        if (!hasBounds)
        {
            Collider[] colliders = agentPrefab.GetComponentsInChildren<Collider>(true);
            foreach (Collider collider in colliders)
            {
                if (collider == null)
                {
                    continue;
                }

                if (collider is BoxCollider)
                {
                    continue;
                }

                Transform colTransform = collider.transform;

                switch (collider)
                {
                    case SphereCollider sphere:
                    {
                        Vector3 center = colTransform.TransformPoint(sphere.center);
                        Vector3 lossyScale = ToAbsVector(colTransform.lossyScale);
                        float maxScale = Mathf.Max(lossyScale.x, lossyScale.y, lossyScale.z);
                        float radius = sphere.radius * Mathf.Max(maxScale, 0.0001f);

                        Vector3 min = center - Vector3.one * radius;
                        Vector3 max = center + Vector3.one * radius;

                        if (!hasBounds)
                        {
                            calculatedBounds = new Bounds(center, Vector3.zero);
                            hasBounds = true;
                        }

                        calculatedBounds.Encapsulate(min);
                        calculatedBounds.Encapsulate(max);
                        break;
                    }
                    case CapsuleCollider capsule:
                    {
                        Vector3 center = colTransform.TransformPoint(capsule.center);
                        Vector3 lossyScale = ToAbsVector(colTransform.lossyScale);
                        float radius = capsule.radius * Mathf.Max(lossyScale.x, lossyScale.y, lossyScale.z);
                        float axisScale = capsule.direction switch
                        {
                            0 => lossyScale.x,
                            1 => lossyScale.y,
                            2 => lossyScale.z,
                            _ => lossyScale.y
                        };
                        float halfAxis = Mathf.Max(0f, capsule.height * axisScale * 0.5f - radius);
                        float effectiveRadius = radius + halfAxis;

                        Vector3 min = center - Vector3.one * effectiveRadius;
                        Vector3 max = center + Vector3.one * effectiveRadius;

                        if (!hasBounds)
                        {
                            calculatedBounds = new Bounds(center, Vector3.zero);
                            hasBounds = true;
                        }

                        calculatedBounds.Encapsulate(min);
                        calculatedBounds.Encapsulate(max);
                        break;
                    }
                    case MeshCollider meshCollider when meshCollider.sharedMesh != null:
                    {
                        Mesh sharedMesh = meshCollider.sharedMesh;
                        Vector3[] vertices = sharedMesh.vertices;

                        foreach (Vector3 vertex in vertices)
                        {
                            Vector3 worldVertex = colTransform.TransformPoint(vertex);
                            if (!hasBounds)
                            {
                                calculatedBounds = new Bounds(worldVertex, Vector3.zero);
                                hasBounds = true;
                            }
                            else
                            {
                                calculatedBounds.Encapsulate(worldVertex);
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (!hasBounds)
        {
            float radius = Mathf.Max(spawnFallbackRadius, 0.1f);
            calculatedBounds = new Bounds(Vector3.zero, Vector3.one * radius * 2f);
        }

        agentPrefabBounds = calculatedBounds;
        agentBoundsInitialized = true;
    }

    private Vector3 GenerateRandomSpawnLocalPosition()
    {
        float angle = Random.Range(0f, Mathf.PI * 2f);
        float distance = Random.Range(0f, Mathf.Max(spawnRange, 0f));
        float height = spawnHeight > 0f ? Random.Range(0f, spawnHeight) : 0f;

        Vector3 horizontal = new Vector3(Mathf.Cos(angle) * distance, 0f, Mathf.Sin(angle) * distance);
        return new Vector3(horizontal.x, height, horizontal.z);
    }

    private bool TryGenerateSpawnPose(out Vector3 worldPosition, out Quaternion worldRotation, GeneticBoidAgent agentToIgnore = null)
    {
        CacheAgentPrefabBounds();

        int attempts = Mathf.Max(1, spawnMaxAttempts);
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            Vector3 localPosition = GenerateRandomSpawnLocalPosition();
            Quaternion localRotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

            worldPosition = transform.TransformPoint(localPosition);
            worldRotation = transform.rotation * localRotation;

            if (!IsSpawnPositionBlocked(worldPosition, worldRotation, agentToIgnore))
            {
                return true;
            }
        }

        worldPosition = default;
        worldRotation = default;
        return false;
    }

    private void ResetAgentTransformsForEvaluation()
    {
        foreach (var agent in simulationAgents)
        {
            if (agent == null)
            {
                continue;
            }

            Vector3 spawnPosition;
            Quaternion spawnRotation;

            if (!TryGenerateSpawnPose(out spawnPosition, out spawnRotation, agent))
            {
                Vector3 fallbackLocal = GenerateRandomSpawnLocalPosition();
                spawnPosition = transform.TransformPoint(fallbackLocal);
                spawnRotation = transform.rotation * Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

                if (enableDetailedLogging)
                {
                    Debug.LogWarning($"[GA] Failed to find collision-free spawn while resetting agents. Using fallback position.");
                }
            }

            agent.ResetTransform(spawnPosition, spawnRotation);
        }
    }

    private bool IsSpawnPositionBlocked(Vector3 worldPosition, Quaternion worldRotation, GeneticBoidAgent agentToIgnore)
    {
        if (spawnCollisionMask == 0)
        {
            return false;
        }

        if (!agentBoundsInitialized || agentPrefabBounds.size == Vector3.zero)
        {
            float radius = Mathf.Max(spawnFallbackRadius, 0.1f);
            var hits = Physics.OverlapSphere(worldPosition, radius, spawnCollisionMask, QueryTriggerInteraction.Ignore);
            return hits.Any(hit => !IsIgnoredSpawnCollider(hit, agentToIgnore));
        }

        Vector3 parentScale = ToAbsVector(transform.lossyScale);
        Vector3 scaledExtents = Vector3.Scale(agentPrefabBounds.extents, parentScale);
        Vector3 scaledCenter = Vector3.Scale(agentPrefabBounds.center, parentScale);
        Vector3 worldCenter = worldPosition + worldRotation * scaledCenter;

        var overlaps = Physics.OverlapBox(worldCenter, scaledExtents, worldRotation, spawnCollisionMask, QueryTriggerInteraction.Ignore);
        return overlaps.Any(hit => !IsIgnoredSpawnCollider(hit, agentToIgnore));
    }

    private bool IsIgnoredSpawnCollider(Collider collider, GeneticBoidAgent agentToIgnore)
    {
        if (collider == null)
        {
            return true;
        }

        if (collider is not BoxCollider)
        {
            return true;
        }

        Transform targetTransform = collider.transform;
        if (targetTransform == transform)
        {
            return true;
        }

        if (agentToIgnore != null && targetTransform.IsChildOf(agentToIgnore.transform))
        {
            return true;
        }

        return false;
    }

    private static Vector3 ToAbsVector(Vector3 value)
    {
        return new Vector3(Mathf.Abs(value.x), Mathf.Abs(value.y), Mathf.Abs(value.z));
    }
    #endregion

    #region 評価フロー
    public void StartEvaluation()
    {
        QueueEvaluationStart();
    }

    private void QueueEvaluationStart()
    {
        pendingEvaluationStart = true;
        EnsureServerHealthCheckStarted();

        if (!serverReady)
        {
            LogWaitingForServer();
            return;
        }

        TryStartEvaluationWhenReady();
    }

    private void TryStartEvaluationWhenReady()
    {
        if (!pendingEvaluationStart || !serverReady)
        {
            return;
        }

        pendingEvaluationStart = false;
        hasLoggedServerWait = false;
        StartGenerationEvaluation();
    }

    private void LogWaitingForServer()
    {
        if (hasLoggedServerWait)
        {
            return;
        }

        hasLoggedServerWait = true;
        LogStatus("Waiting for Python server health before starting evaluation...");
    }

    private void EnsureServerHealthCheckStarted()
    {
        if (serverReady || serverHealthCoroutine != null)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(serverHealthUrl))
        {
            serverReady = true;
            LogStatus("Server health URL not set. Skipping health check.");
            TryStartEvaluationWhenReady();
            return;
        }

        serverHealthCoroutine = StartCoroutine(ServerHealthCheckRoutine());
    }

    public void StartGenerationEvaluation()
    {
        if (!serverReady)
        {
            pendingEvaluationStart = true;
            EnsureServerHealthCheckStarted();
            LogWaitingForServer();
            return;
        }

        pendingEvaluationStart = false;
        hasLoggedServerWait = false;

        EnsureSimulationAgentsReady();

        if (simulationAgents.Count == 0 || individuals.Count == 0)
        {
            Debug.LogWarning("[GA] Cannot start evaluation: agents or individuals missing.");
            return;
        }

        if (generationInProgress)
        {
            if (enableDetailedLogging)
            {
                Debug.LogWarning("[GA] Evaluation already in progress.");
            }
            return;
        }

        generationInProgress = true;
        currentIndividualIndex = 0;
        LogStatus($"Starting generation {currentGeneration}, evaluating {individuals.Count} individuals.");
        PrepareIndividualEvaluation(currentIndividualIndex);
    }

    private IEnumerator ServerHealthCheckRoutine()
    {
        bool isFirstAttempt = true;
        float retryInterval = Mathf.Max(0.1f, serverHealthRetryIntervalSeconds);
        WaitForSeconds waitInstruction = new WaitForSeconds(retryInterval);

        while (!serverReady)
        {
            if (!isFirstAttempt)
            {
                yield return waitInstruction;
            }
            else
            {
                isFirstAttempt = false;
            }

            using (UnityWebRequest request = UnityWebRequest.Get(serverHealthUrl))
            {
                request.timeout = Mathf.Max(1, serverHealthRequestTimeoutSeconds);
                yield return request.SendWebRequest();

                if (IsUnityWebRequestSuccessful(request))
                {
                    serverReady = true;
                    LogStatus($"Server health check succeeded ({serverHealthUrl}).");
                    break;
                }

                string errorDetails = string.IsNullOrEmpty(request.error) ? "Unknown error" : request.error;
                long responseCode = request.responseCode;
                string responseLabel = responseCode > 0 ? responseCode.ToString() : "no response";
                string retryMessage = $"[GA] Server health check failed ({responseLabel}): {errorDetails}. Retrying in {retryInterval:F1}s.";

                if (hasLoggedServerWait)
                {
                    Debug.LogWarning(retryMessage);
                }
                else if (enableDetailedLogging)
                {
                    Debug.Log(retryMessage);
                }
            }
        }

        serverHealthCoroutine = null;
        TryStartEvaluationWhenReady();
    }

    private static bool IsUnityWebRequestSuccessful(UnityWebRequest request)
    {
        if (request == null)
        {
            return false;
        }

        return request.responseCode >= 200 && request.responseCode < 300;
    }

    private void PrepareIndividualEvaluation(int index)
    {
        if (index < 0 || index >= individuals.Count)
        {
            Debug.LogError($"[GA] Invalid individual index {index}");
            generationInProgress = false;
            waitingForServerScore = false;
            return;
        }

        if (videoRecorder == null)
        {
            videoRecorder = FindObjectOfType<VideoRecorder>();
        }

        videoRecorder?.SetEvaluationContext(currentGeneration, index);

        var genes = individuals[index].Genes;

        ResetAgentTransformsForEvaluation();

        foreach (var agent in simulationAgents)
        {
            if (agent == null) continue;
            agent.SetGenes(genes);
            agent.ResetFitness();
        }

        OnIndividualEvaluationStarted?.Invoke(currentGeneration, index);

        waitingForServerScore = true;
        BeginServerScoreTimeoutWatch(index);
        LogStatus($"Individual {index + 1}/{individuals.Count} applied. Awaiting server score...");

        if (enableDetailedLogging)
        {
            Debug.Log($"[GA] Generation {currentGeneration}, Individual {index + 1}/{individuals.Count}: genes applied, awaiting server score.");
        }

        if (videoRecorder == null)
        {
            videoRecorder = FindObjectOfType<VideoRecorder>();
            if (videoRecorder == null)
            {
                Debug.LogWarning("[GA] VideoRecorder not found. Using local fitness estimate.");
                waitingForServerScore = false;
                CancelServerScoreTimeoutWatch();
                individuals[index].Fitness = CalculateFallbackFitness();
                LogStatus($"Individual {index + 1} fallback fitness = {individuals[index].Fitness:F3}");
                AdvanceToNextIndividual();
            }
        }
    }

    private float CalculateFallbackFitness()
    {
        if (simulationAgents.Count == 0)
        {
            return 0f;
        }

        float total = 0f;
        int count = 0;
        foreach (var agent in simulationAgents)
        {
            if (agent == null) continue;
            total += Mathf.Max(0f, agent.fitness);
            count++;
        }
        return count > 0 ? total / count : 0f;
    }

    public void HandleServerScoreReceived(int episodeNumber, float score)
    {
        if (!generationInProgress || !waitingForServerScore)
        {
            return;
        }

        if (currentIndividualIndex < 0 || currentIndividualIndex >= individuals.Count)
        {
            Debug.LogWarning("[GA] Received score but individual index is out of range.");
            return;
        }

        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();
        float clampedScore = Mathf.Max(0f, score);
        individuals[currentIndividualIndex].Fitness = clampedScore;

        LogStatus($"Received server score {clampedScore:F3} for individual {currentIndividualIndex + 1}.");

        if (enableDetailedLogging)
        {
            Debug.Log($"[GA] Generation {currentGeneration}, Individual {currentIndividualIndex + 1}: server score = {clampedScore:F4}");
        }

        AdvanceToNextIndividual();
    }

    private void AdvanceToNextIndividual()
    {
        currentIndividualIndex++;

        if (currentIndividualIndex >= individuals.Count)
        {
            CompleteGeneration();
        }
        else
        {
            PrepareIndividualEvaluation(currentIndividualIndex);
        }
    }

    private void CompleteGeneration()
    {
        generationInProgress = false;
        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();

        CalculateStatistics();
        WriteGenerationCsvRow();

        if (enableDetailedLogging)
        {
            Debug.Log($"[GA] Generation {currentGeneration}: Best Fitness = {bestFitness:F4}, Avg Fitness = {avgFitness:F4}");
        }
        else
        {
            LogStatus($"Generation {currentGeneration} complete. Best {bestFitness:F3}, Avg {avgFitness:F3}.");
        }

        if (logBestGenes)
        {
            Debug.Log("[GA] Best Genes: " + string.Join(", ", bestGenes.Select((g, i) => $"{GeneticBoidAgent.GeneNames[i]}={g:F2}")));
        }

        if (maxGenerations > 0 && currentGeneration + 1 >= maxGenerations)
        {
            Debug.Log($"[GA] Reached max generations ({maxGenerations}). Evolution complete.");
            return;
        }

        var nextPopulation = CreateNextGeneration();
        individuals.Clear();
        individuals.AddRange(nextPopulation);
        currentGeneration++;
        WritePopulationParametersToCsv(currentGeneration, individuals);

        if (autoRun)
        {
            QueueEvaluationStart();
        }
    }
    #endregion

    #region 選択・交叉・突然変異
    private List<Individual> CreateNextGeneration()
    {
        var ordered = individuals.OrderByDescending(ind => ind.Fitness).ToList();
        var newPopulation = new List<Individual>();

        for (int i = 0; i < eliteCount && i < ordered.Count; i++)
        {
            newPopulation.Add(new Individual
            {
                Genes = CloneGenes(ordered[i].Genes),
                Fitness = 0f
            });
        }

        float totalFitness = Mathf.Max(0.0001f, ordered.Sum(ind => Mathf.Max(0f, ind.Fitness)));

        while (newPopulation.Count < populationSize)
        {
            var parent1 = SelectParent(totalFitness);
            var parent2 = SelectParent(totalFitness);

            var (childGenes1, childGenes2) = PerformCrossover(parent1.Genes, parent2.Genes);

            MutateGenes(childGenes1);
            MutateGenes(childGenes2);

            newPopulation.Add(new Individual { Genes = childGenes1, Fitness = 0f });
            if (newPopulation.Count < populationSize)
            {
                newPopulation.Add(new Individual { Genes = childGenes2, Fitness = 0f });
            }
        }

        return newPopulation;
    }

    private Individual SelectParent(float totalFitness)
    {
        if (totalFitness <= 0f)
        {
            return individuals[Random.Range(0, individuals.Count)];
        }

        float pick = Random.Range(0f, totalFitness);
        float cumulative = 0f;
        foreach (var individual in individuals)
        {
            cumulative += Mathf.Max(0f, individual.Fitness);
            if (cumulative >= pick)
            {
                return individual;
            }
        }

        return individuals[^1];
    }

    private (float[], float[]) PerformCrossover(float[] parent1, float[] parent2)
    {
        if (Random.value >= crossoverRate)
        {
            return (CloneGenes(parent1), CloneGenes(parent2));
        }

        return BLXAlphaCrossover(parent1, parent2, 0.5f);
    }

    private static (float[], float[]) BLXAlphaCrossover(float[] parent1, float[] parent2, float alpha)
    {
        int geneCount = GeneticBoidAgent.GeneCount;
        float[] child1 = new float[geneCount];
        float[] child2 = new float[geneCount];

        for (int i = 0; i < geneCount; i++)
        {
            float min = Mathf.Min(parent1[i], parent2[i]);
            float max = Mathf.Max(parent1[i], parent2[i]);
            float range = max - min;

            float lowerBound = min - alpha * range;
            float upperBound = max + alpha * range;

            child1[i] = GeneticBoidAgent.ClampGene(i, Random.Range(lowerBound, upperBound));
            child2[i] = GeneticBoidAgent.ClampGene(i, Random.Range(lowerBound, upperBound));
        }

        return (child1, child2);
    }

    private void MutateGenes(float[] genes)
    {
        if (Random.value > mutationRate)
        {
            return;
        }

        for (int i = 0; i < genes.Length; i++)
        {
            if (Random.value < 0.5f)
            {
                float range = GeneticBoidAgent.GeneMaxs[i] - GeneticBoidAgent.GeneMins[i];
                float mutation = GaussianRandom() * mutationStrength * range;
                genes[i] = GeneticBoidAgent.ClampGene(i, genes[i] + mutation);
            }
        }
    }

    private static float[] CloneGenes(float[] source)
    {
        float[] clone = new float[source.Length];
        source.CopyTo(clone, 0);
        return clone;
    }

    private float GaussianRandom()
    {
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);
    }
    #endregion

    #region 統計
    private void CalculateStatistics()
    {
        if (individuals.Count == 0)
        {
            bestFitness = 0f;
            avgFitness = 0f;
            bestGenes = new float[GeneticBoidAgent.GeneCount];
            return;
        }

        var ordered = individuals.OrderByDescending(ind => ind.Fitness).ToList();
        bestFitness = ordered[0].Fitness;
        avgFitness = individuals.Average(ind => ind.Fitness);
        bestGenes = CloneGenes(ordered[0].Genes);
    }
    #endregion

    #region CSV ログ出力
    private void InitializeCsvLogging()
    {
        csvHeaderWritten = false;
        populationCsvHeaderWritten = false;

        if (!enableCsvLogging)
        {
            csvFilePath = null;
            populationCsvFilePath = null;
            return;
        }

        string sanitizedFileName = string.IsNullOrWhiteSpace(csvFileName) ? "ga_progress.csv" : csvFileName.Trim();
        string populationFile = string.IsNullOrWhiteSpace(populationCsvFileName) ? "ga_population.csv" : populationCsvFileName.Trim();
        string logsDirectory = Path.Combine(Application.dataPath, "..", "Logs");

        try
        {
            Directory.CreateDirectory(logsDirectory);
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[GA] Failed to create Logs directory: {ex.Message}");
            enableCsvLogging = false;
            csvFilePath = null;
            return;
        }

        InitializeCsvFile(logsDirectory, sanitizedFileName, CreateCsvHeader(), ref csvFilePath, ref csvHeaderWritten);
        InitializeCsvFile(logsDirectory, populationFile, CreatePopulationCsvHeader(), ref populationCsvFilePath, ref populationCsvHeaderWritten);
    }

    private string CreateCsvHeader()
    {
        var sb = new StringBuilder();
        sb.Append("Generation,BestFitness,AverageFitness");

        for (int i = 0; i < GeneticBoidAgent.GeneCount; i++)
        {
            sb.Append(',');
            sb.Append(GeneticBoidAgent.GeneNames[i]);
        }

        return sb.ToString();
    }

    private string CreatePopulationCsvHeader()
    {
        var sb = new StringBuilder();
        sb.Append("Generation,Individual");

        for (int i = 0; i < GeneticBoidAgent.GeneCount; i++)
        {
            sb.Append(',');
            sb.Append(GeneticBoidAgent.GeneNames[i]);
        }

        return sb.ToString();
    }

    private void InitializeCsvFile(string directory, string fileName, string header, ref string filePath, ref bool headerWritten)
    {
        filePath = Path.GetFullPath(Path.Combine(directory, fileName));
        bool fileExists = File.Exists(filePath);
        bool shouldOverwrite = overwriteCsvOnStart || !fileExists;

        if (!shouldOverwrite && fileExists)
        {
            try
            {
                shouldOverwrite = new FileInfo(filePath).Length == 0;
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[GA] Unable to inspect CSV file '{fileName}'. Will append without rewriting header. {ex.Message}");
            }
        }

        if (shouldOverwrite)
        {
            try
            {
                using StreamWriter writer = new StreamWriter(filePath, false, Encoding.UTF8);
                writer.WriteLine(header);
                headerWritten = true;
                LogStatus($"CSV logging initialized: {filePath}");
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[GA] Failed to initialize CSV logging for '{fileName}': {ex.Message}");
                filePath = null;
                headerWritten = false;
            }
        }
        else
        {
            headerWritten = true;
            LogStatus($"CSV logging appending to existing file: {filePath}");
        }
    }

    private void WriteGenerationCsvRow()
    {
        if (!enableCsvLogging || string.IsNullOrEmpty(csvFilePath))
        {
            return;
        }

        try
        {
            if (!csvHeaderWritten)
            {
                using StreamWriter headerWriter = new StreamWriter(csvFilePath, false, Encoding.UTF8);
                headerWriter.WriteLine(CreateCsvHeader());
                csvHeaderWritten = true;
            }

            int generationNumber = currentGeneration + 1;
            var sb = new StringBuilder();
            sb.Append(generationNumber);
            sb.Append(',');
            sb.Append(bestFitness.ToString("F4", CultureInfo.InvariantCulture));
            sb.Append(',');
            sb.Append(avgFitness.ToString("F4", CultureInfo.InvariantCulture));

            for (int i = 0; i < bestGenes.Length; i++)
            {
                sb.Append(',');
                sb.Append(bestGenes[i].ToString("F4", CultureInfo.InvariantCulture));
            }

            sb.AppendLine();
            File.AppendAllText(csvFilePath, sb.ToString(), Encoding.UTF8);
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[GA] Failed to write CSV row: {ex.Message}");
        }
    }

    private void WritePopulationParametersToCsv(int generationIndex, List<Individual> population)
    {
        if (!enableCsvLogging || string.IsNullOrEmpty(populationCsvFilePath))
        {
            return;
        }

        try
        {
            if (!populationCsvHeaderWritten)
            {
                using StreamWriter headerWriter = new StreamWriter(populationCsvFilePath, false, Encoding.UTF8);
                headerWriter.WriteLine(CreatePopulationCsvHeader());
                populationCsvHeaderWritten = true;
            }

            int generationNumber = generationIndex + 1;
            using StreamWriter writer = new StreamWriter(populationCsvFilePath, true, Encoding.UTF8);
            for (int i = 0; i < population.Count; i++)
            {
                var genes = population[i].Genes;
                if (genes == null)
                {
                    genes = new float[GeneticBoidAgent.GeneCount];
                }

                var sb = new StringBuilder();
                sb.Append(generationNumber);
                sb.Append(',');
                sb.Append(i + 1);

                for (int geneIndex = 0; geneIndex < GeneticBoidAgent.GeneCount; geneIndex++)
                {
                    float geneValue = geneIndex < genes.Length ? genes[geneIndex] : 0f;
                    sb.Append(',');
                    sb.Append(geneValue.ToString("F4", CultureInfo.InvariantCulture));
                }

                writer.WriteLine(sb.ToString());
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[GA] Failed to write population CSV rows: {ex.Message}");
        }
    }
    #endregion

    #region 手動制御 API
    public void StopEvaluation()
    {
        generationInProgress = false;
        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();
        pendingEvaluationStart = false;
        hasLoggedServerWait = false;
        autoRun = false;
    }

    public void ResetGA()
    {
        currentGeneration = 0;
        generationInProgress = false;
        waitingForServerScore = false;
        CancelServerScoreTimeoutWatch();
        pendingEvaluationStart = false;
        hasLoggedServerWait = false;
        InitializePopulationGenes();
        InitializeCsvLogging();
        WritePopulationParametersToCsv(currentGeneration, individuals);
        ResetAgentTransformsForEvaluation();

        if (autoRun)
        {
            QueueEvaluationStart();
        }
    }

    private void BeginServerScoreTimeoutWatch(int expectedIndividualIndex)
    {
        CancelServerScoreTimeoutWatch();

        if (serverScoreTimeoutSeconds <= 0f)
        {
            return;
        }

        serverScoreTimeoutCoroutine = StartCoroutine(ServerScoreTimeoutRoutine(expectedIndividualIndex));
    }

    private void CancelServerScoreTimeoutWatch()
    {
        if (serverScoreTimeoutCoroutine == null)
        {
            return;
        }

        StopCoroutine(serverScoreTimeoutCoroutine);
        serverScoreTimeoutCoroutine = null;
    }

    private IEnumerator ServerScoreTimeoutRoutine(int expectedIndividualIndex)
    {
        yield return new WaitForSeconds(serverScoreTimeoutSeconds);
        serverScoreTimeoutCoroutine = null;
        HandleServerScoreTimeout(expectedIndividualIndex);
    }

    private void HandleServerScoreTimeout(int expectedIndividualIndex)
    {
        if (!generationInProgress || !waitingForServerScore)
        {
            return;
        }

        if (currentIndividualIndex != expectedIndividualIndex)
        {
            return;
        }

        waitingForServerScore = false;
        float fallbackFitness = CalculateFallbackFitness();
        individuals[currentIndividualIndex].Fitness = fallbackFitness;

        Debug.LogWarning($"[GA] Timed out waiting for server score after {serverScoreTimeoutSeconds:F1}s at individual {currentIndividualIndex + 1}. Using fallback fitness {fallbackFitness:F3}.");
        LogStatus($"Timeout fallback fitness {fallbackFitness:F3} applied to individual {currentIndividualIndex + 1}.");

        AdvanceToNextIndividual();
    }
    #endregion

}
