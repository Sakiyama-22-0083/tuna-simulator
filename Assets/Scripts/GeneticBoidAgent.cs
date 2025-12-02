using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 遺伝的アルゴリズムによってBoidsパラメータを最適化するエージェント
/// 各個体は複数の連続値遺伝子（重み・速度・半径など）を持ち、適応度に基づいて進化する
/// </summary>
public class GeneticBoidAgent : MonoBehaviour
{
    #region コンポーネント参照
    private Rigidbody rBody;
    #endregion

    #region 遺伝子（最適化対象パラメータ）
    [Header("遺伝子（Boidsパラメータ）")]
    public float separationWeight = 1.0f;      // 分離の重み
    public float alignmentWeight = 1.0f;       // 整列の重み
    public float cohesionWeight = 1.0f;        // 結合の重み
    public float obstacleAvoidWeight = 2.0f;   // 障害物回避の重み
    public float baseSpeed = 2f;               // 前進速度
    public float separationRadius = 2f;        // 近距離反発半径（InnerRadius）

    [Header("固定設定")]
    [Tooltip("近傍探索半径（遺伝的アルゴリズムの調整対象外）")]
    public float detectionRadius = 5f;         // 周囲探索半径

    public const int GeneCount = 6;
    public static readonly string[] GeneNames =
    {
        "Separation", "Alignment", "Cohesion", "Obstacle", "MoveSpeed", "InnerRadius"
    };

    // 遺伝子ごとの下限・上限
    public static readonly float[] GeneMins = { 0f, 0f, 0f, 0f, 0.5f, 0.5f };
    public static readonly float[] GeneMaxs = { 3f, 3f, 3f, 3f, 5f, 5f };

    public static float ClampGene(int index, float value)
    {
        return Mathf.Clamp(value, GeneMins[index], GeneMaxs[index]);
    }

    private static float RandomGene(int index)
    {
        return Random.Range(GeneMins[index], GeneMaxs[index]);
    }
    #endregion

    #region Boidsパラメータ（固定）
    [Header("Boidsパラメータ設定")]
    [Tooltip("最大回転速度")]
    public float maxRotationSpeed = 2f;
    #endregion

    #region 検出・判定設定
    [Header("検出設定")]
    public LayerMask agentLayer;
    public LayerMask obstacleLayer;

    [SerializeField, Tooltip("非割り当てメモリアロケーションなしで検出に使用するバッファのサイズ。最大検出数に合わせて調整してください。")]
    [Min(8)] private int detectionBufferCapacity = 128;

    [Header("近傍選択設定")]
    [SerializeField, Min(1)] private int maxAgentsConsidered = 10;

    private readonly List<GeneticBoidAgent> nearbyAgents = new();
    private readonly List<Collider> nearbyObstacles = new();
    private readonly List<GeneticBoidAgent> nearestAgentsBuffer = new();
    private Collider[] agentColliderBuffer;
    private Collider[] obstacleColliderBuffer;
    private bool agentBufferOverflowLogged;
    private bool obstacleBufferOverflowLogged;
    #endregion

    #region 適応度評価
    [Header("適応度評価")]
    public float fitness = 0f;
    
    private float totalGroupCohesion = 0f;      // 群れのまとまり度
    private float totalSmoothness = 0f;         // 動きの滑らかさ
    private int collisionCount = 0;             // 衝突回数
    private float totalSpeed = 0f;              // 累積速度
    private int evaluationSteps = 0;            // 評価ステップ数
    #endregion

    #region 初期化
    private void Awake()
    {
        EnsureRigidbody();
    }

    void Start()
    {
        EnsureRigidbody();
        ResetFitness();
    }

    private void EnsureRigidbody()
    {
        if (rBody == null)
        {
            rBody = GetComponent<Rigidbody>();
        }

        if (rBody == null)
        {
            Debug.LogError("[GeneticBoidAgent] Rigidbody component is required.", this);
        }
    }

    private void OnValidate()
    {
        if (detectionBufferCapacity < maxAgentsConsidered)
        {
            detectionBufferCapacity = maxAgentsConsidered;
        }
    }

    /// <summary>
    /// 遺伝子をランダム初期化
    /// </summary>
    public void RandomizeGenes()
    {
        var genes = GenerateRandomGenes();
        ApplyGeneValues(genes);
    }

    /// <summary>
    /// 遺伝子を設定
    /// </summary>
    public void SetGenes(float[] genes)
    {
        if (genes == null || genes.Length != GeneCount)
        {
            Debug.LogWarning("[GeneticBoidAgent] Invalid gene array supplied.");
            return;
        }

        ApplyGeneValues(genes);
    }

    /// <summary>
    /// 遺伝子を取得
    /// </summary>
    public float[] GetGenes()
    {
        return new float[]
        {
            separationWeight,
            alignmentWeight,
            cohesionWeight,
            obstacleAvoidWeight,
            baseSpeed,
            separationRadius
        };
    }

    public static float[] GenerateRandomGenes()
    {
        var genes = new float[GeneCount];
        for (int i = 0; i < GeneCount; i++)
        {
            genes[i] = RandomGene(i);
        }
        return genes;
    }

    private void ApplyGeneValues(float[] genes)
    {
        separationWeight = ClampGene(0, genes[0]);
        alignmentWeight = ClampGene(1, genes[1]);
        cohesionWeight = ClampGene(2, genes[2]);
        obstacleAvoidWeight = ClampGene(3, genes[3]);
        baseSpeed = ClampGene(4, genes[4]);
        separationRadius = ClampGene(5, genes[5]);
    }
    #endregion

    #region 物理更新
    void FixedUpdate()
    {
        // 近くのオブジェクトを検出
        DetectNearbyObjects();

        // Boidsアルゴリズムを実行
        Vector3 separation = CalculateSeparation() * separationWeight;

        Vector3 alignment = CalculateAlignment() * alignmentWeight;
        Vector3 cohesion = CalculateCohesion() * cohesionWeight;
        Vector3 avoidance = CalculateObstacleAvoidance() * obstacleAvoidWeight;

        Vector3 targetDirection = transform.forward;

        if (nearbyAgents.Count > 0 || nearbyObstacles.Count > 0)
        {
            targetDirection = separation + alignment + cohesion + avoidance;
            targetDirection += transform.forward * Mathf.Max(targetDirection.magnitude, 1f) * 0.5f;
        }

        targetDirection.y = 0f;

        if (targetDirection.sqrMagnitude < 0.0001f)
        {
            targetDirection = transform.forward;
            targetDirection.y = 0f;
        }

        if (targetDirection.sqrMagnitude > 0.0001f)
        {
            targetDirection.Normalize();
            Quaternion targetRotation = Quaternion.LookRotation(targetDirection, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, maxRotationSpeed * Time.fixedDeltaTime);
        }

        rBody.velocity = transform.forward * baseSpeed;

        // 適応度を評価
        EvaluateFitness();
    }
    #endregion

    #region Boidsアルゴリズム実装
    /// <summary>
    /// 近くのオブジェクトを検出
    /// </summary>
    private void DetectNearbyObjects()
    {
        nearbyAgents.Clear();
        nearbyObstacles.Clear();
        EnsureDetectionBuffers();

        int agentHits = Physics.OverlapSphereNonAlloc(transform.position, detectionRadius, agentColliderBuffer, agentLayer, QueryTriggerInteraction.Ignore);
        if (agentHits >= agentColliderBuffer.Length && !agentBufferOverflowLogged)
        {
            Debug.LogWarning($"[GeneticBoidAgent] Agent detection buffer ({agentColliderBuffer.Length}) saturated. Increase detectionBufferCapacity to capture all neighbors.");
            agentBufferOverflowLogged = true;
        }

        for (int i = 0; i < agentHits && i < agentColliderBuffer.Length; i++)
        {
            Collider col = agentColliderBuffer[i];
            if (col == null || col.gameObject == gameObject)
            {
                continue;
            }

            if (col.TryGetComponent(out GeneticBoidAgent agent) && agent != this)
            {
                nearbyAgents.Add(agent);
            }
        }

        int obstacleHits = Physics.OverlapSphereNonAlloc(transform.position, detectionRadius, obstacleColliderBuffer, obstacleLayer, QueryTriggerInteraction.Ignore);
        if (obstacleHits >= obstacleColliderBuffer.Length && !obstacleBufferOverflowLogged)
        {
            Debug.LogWarning($"[GeneticBoidAgent] Obstacle detection buffer ({obstacleColliderBuffer.Length}) saturated. Increase detectionBufferCapacity to capture all obstacles.");
            obstacleBufferOverflowLogged = true;
        }

        for (int i = 0; i < obstacleHits && i < obstacleColliderBuffer.Length; i++)
        {
            Collider col = obstacleColliderBuffer[i];
            if (col != null)
            {
                nearbyObstacles.Add(col);
            }
        }
    }

    /// <summary>
    /// 分離：近くのエージェントから離れる
    /// </summary>
    private Vector3 CalculateSeparation()
    {
        Vector3 force = Vector3.zero;
        int count = 0;

        var neighbours = GetNearestAgents();

        foreach (GeneticBoidAgent agent in neighbours)
        {
            float distance = Vector3.Distance(transform.position, agent.transform.position);
            if (distance < separationRadius && distance > 0.001f)
            {
                Vector3 diff = transform.position - agent.transform.position;
                force += diff.normalized / distance;
                count++;
            }
        }

        if (count > 0)
        {
            force /= count;
        }

        if (force.sqrMagnitude < 0.0001f)
        {
            return Vector3.zero;
        }

        return force.normalized;
    }

    /// <summary>
    /// 整列：近くのエージェントと速度を合わせる
    /// </summary>
    private Vector3 CalculateAlignment()
    {
        if (rBody == null)
        {
            return Vector3.zero;
        }

        Vector3 avgVelocity = Vector3.zero;
        int count = 0;

        var neighbours = GetNearestAgents();

        foreach (GeneticBoidAgent agent in neighbours)
        {
            if (agent == null || agent.rBody == null)
            {
                continue;
            }

            avgVelocity += agent.rBody.velocity;
            count++;
        }

        if (count > 0)
        {
            avgVelocity /= count;
            Vector3 desired = avgVelocity - rBody.velocity;
            desired.y = 0f;
            if (desired.sqrMagnitude < 0.0001f)
            {
                return Vector3.zero;
            }
            return desired.normalized;
        }

        return Vector3.zero;
    }

    private List<GeneticBoidAgent> GetNearestAgents()
    {
        nearestAgentsBuffer.Clear();

        if (nearbyAgents.Count == 0)
        {
            return nearestAgentsBuffer;
        }

        for (int i = 0; i < nearbyAgents.Count; i++)
        {
            GeneticBoidAgent agent = nearbyAgents[i];
            if (agent == null || agent == this)
            {
                continue;
            }
            nearestAgentsBuffer.Add(agent);
        }

        if (nearestAgentsBuffer.Count <= 1)
        {
            return nearestAgentsBuffer;
        }

        int targetCount = Mathf.Min(maxAgentsConsidered, nearestAgentsBuffer.Count);
        for (int i = 0; i < targetCount; i++)
        {
            int bestIndex = i;
            float bestDistance = DistanceSquaredTo(nearestAgentsBuffer[i]);

            for (int j = i + 1; j < nearestAgentsBuffer.Count; j++)
            {
                float candidateDistance = DistanceSquaredTo(nearestAgentsBuffer[j]);
                if (candidateDistance < bestDistance)
                {
                    bestDistance = candidateDistance;
                    bestIndex = j;
                }
            }

            if (bestIndex != i)
            {
                (nearestAgentsBuffer[i], nearestAgentsBuffer[bestIndex]) = (nearestAgentsBuffer[bestIndex], nearestAgentsBuffer[i]);
            }
        }

        if (nearestAgentsBuffer.Count > maxAgentsConsidered)
        {
            nearestAgentsBuffer.RemoveRange(maxAgentsConsidered, nearestAgentsBuffer.Count - maxAgentsConsidered);
        }

        return nearestAgentsBuffer;
    }

    /// <summary>
    /// 結合：群れの中心に向かう
    /// </summary>
    private Vector3 CalculateCohesion()
    {
        Vector3 centerOfMass = Vector3.zero;
        int count = 0;

        var neighbours = GetNearestAgents();

        foreach (GeneticBoidAgent agent in neighbours)
        {
            float distance = Vector3.Distance(transform.position, agent.transform.position);
            if (distance > separationRadius)
            {
                centerOfMass += agent.transform.position;
                count++;
            }
        }

        if (count > 0)
        {
            centerOfMass /= count;
            Vector3 towardsCenter = centerOfMass - transform.position;
            towardsCenter.y = 0f;
            if (towardsCenter.sqrMagnitude < 0.0001f)
            {
                return Vector3.zero;
            }
            return towardsCenter.normalized;
        }

        return Vector3.zero;
    }

    /// <summary>
    /// 障害物回避
    /// </summary>
    private Vector3 CalculateObstacleAvoidance()
    {
        Vector3 avoidForce = Vector3.zero;
        int count = 0;

        foreach (Collider obstacle in nearbyObstacles)
        {
            if (obstacle == null) continue;

            Vector3 closestPoint;
            if (obstacle is MeshCollider meshCollider && !meshCollider.convex)
            {
                closestPoint = obstacle.bounds.ClosestPoint(transform.position);
            }
            else
            {
                closestPoint = obstacle.ClosestPoint(transform.position);
            }
            float distance = Vector3.Distance(transform.position, closestPoint);

            if (distance < separationRadius && distance > 0.001f)
            {
                Vector3 diff = transform.position - closestPoint;
                avoidForce += diff.normalized / distance;
                count++;
            }
        }

        if (count > 0)
        {
            avoidForce /= count;
        }

        if (avoidForce.sqrMagnitude < 0.0001f)
        {
            return Vector3.zero;
        }

        return avoidForce.normalized;
    }
    #endregion

    #region 適応度評価
    /// <summary>
    /// 適応度をリセット
    /// </summary>
    public void ResetFitness()
    {
        fitness = 0f;
        totalGroupCohesion = 0f;
        totalSmoothness = 0f;
        collisionCount = 0;
        totalSpeed = 0f;
        evaluationSteps = 0;
    }

    /// <summary>
    /// 各ステップで適応度を評価
    /// </summary>
    private void EvaluateFitness()
    {
        evaluationSteps++;

        // 1. 群れのまとまり度（近くにいるエージェントの数）
        float cohesionScore = nearbyAgents.Count / 10f; // 最大10体を理想とする
        cohesionScore = Mathf.Min(cohesionScore, 1f);
        totalGroupCohesion += cohesionScore;

        // 2. 動きの滑らかさ（角速度が小さいほど良い）
        float smoothness = 1f - Mathf.Min(rBody.angularVelocity.magnitude / 5f, 1f);
        totalSmoothness += smoothness;

        // 3. 速度の安定性
        float speedStability = 1f - Mathf.Abs(rBody.velocity.magnitude - baseSpeed) / baseSpeed;
        speedStability = Mathf.Max(speedStability, 0f);
        totalSpeed += speedStability;

        // 適応度を計算（各指標の平均）
        if (evaluationSteps > 0)
        {
            float avgCohesion = totalGroupCohesion / evaluationSteps;
            float avgSmoothness = totalSmoothness / evaluationSteps;
            float avgSpeed = totalSpeed / evaluationSteps;
            float collisionPenalty = collisionCount * 0.1f;

            fitness = (avgCohesion * 0.4f + avgSmoothness * 0.3f + avgSpeed * 0.3f) - collisionPenalty;
            fitness = Mathf.Max(fitness, 0f);
        }
    }

    /// <summary>
    /// 衝突検出
    /// </summary>
    private void OnCollisionEnter(Collision collision)
    {
        // 障害物との衝突をカウント
        if (((1 << collision.gameObject.layer) & obstacleLayer) != 0)
        {
            collisionCount++;
        }
    }
    #endregion

    private void EnsureDetectionBuffers()
    {
        int requiredCapacity = Mathf.Max(maxAgentsConsidered, detectionBufferCapacity);

        if (agentColliderBuffer == null || agentColliderBuffer.Length != requiredCapacity)
        {
            agentColliderBuffer = new Collider[requiredCapacity];
            agentBufferOverflowLogged = false;
        }

        if (obstacleColliderBuffer == null || obstacleColliderBuffer.Length != requiredCapacity)
        {
            obstacleColliderBuffer = new Collider[requiredCapacity];
            obstacleBufferOverflowLogged = false;
        }
    }

    private float DistanceSquaredTo(GeneticBoidAgent agent)
    {
        if (agent == null)
        {
            return float.PositiveInfinity;
        }

        Vector3 offset = agent.transform.position - transform.position;
        return offset.sqrMagnitude;
    }

    public void ResetTransform(Vector3 worldPosition, Quaternion worldRotation)
    {
        EnsureRigidbody();
        rBody.angularVelocity = Vector3.zero;
        rBody.velocity = Vector3.zero;
        transform.SetPositionAndRotation(worldPosition, worldRotation);
    }

    #region 位置リセット
    /// <summary>
    /// エージェントの位置と速度をリセット
    /// </summary>
    public void ResetPosition()
    {
        EnsureRigidbody();
        rBody.angularVelocity = Vector3.zero;
        rBody.velocity = Vector3.zero;

        transform.localPosition = new Vector3(
            Random.Range(-10f, 10f),
            Random.Range(0f, 5f),
            Random.Range(-10f, 10f)
        );
        transform.localRotation = Quaternion.Euler(
            Random.Range(-10f, 10f),
            Random.Range(0f, 360f),
            Random.Range(-10f, 10f)
        );
    }
    #endregion

    #region デバッグ可視化
    private void OnDrawGizmosSelected()
    {
        // 検出範囲
        Gizmos.color = Color.green;
        Gizmos.DrawWireSphere(transform.position, detectionRadius);

        // 分離範囲
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(transform.position, separationRadius);

        // 近くのエージェントへの線
        Gizmos.color = Color.yellow;
        foreach (GeneticBoidAgent agent in nearbyAgents)
        {
            if (agent != null)
            {
                Gizmos.DrawLine(transform.position, agent.transform.position);
            }
        }
    }
    #endregion
}
