using UnityEngine;

/// <summary>
/// リーダークラス
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class Leader : Boid
{
    /// <summary>
    /// 他のエージェントから距離をとるメソッド
    /// </summary>
    /// <returns>分離ベクトル</returns>
    public override Vector3 Separation()
    {
        Vector3 vector = new(0, 0, 0);
        int innerAgentNum = 0;

        foreach (var agent in outerList)
        {
            var distance = (transform.position - agent.transform.position).magnitude;

            if (distance <= innerRadius)
            {
                vector += (transform.position - agent.transform.position).normalized * innerRadius / distance;
                innerAgentNum++;
            }
        }

        if (innerAgentNum > 0)
        {
            vector /= innerAgentNum;
        }
        vector += Avoid();

        return vector;
    }


    /// <summary>
    /// 視界内のエージェントと向きを合わせるメソッド
    /// </summary>
    /// <returns>整列ベクトル</returns>
    public override Vector3 Align()
    {
        Vector3 vector = new();

        foreach (var agent in outerList)
        {
            vector += agent.GetVelocity;
        }

        // カメラ外に移動しないように上下方向を向かないようにする
        vector.y = 0;

        return vector.normalized;
    }

    /// <summary>
    /// 視界内のエージェントの中心座標を目指すメソッド
    /// </summary>
    /// <returns>結合ベクトル</returns>
    public override Vector3 Cohesion()
    {
        Vector3 totalPosition = new(0, 0, 0);

        foreach (var agent in outerList)
        {
            var targetPosition = agent.gameObject.transform.position;

            if (Vector3.Distance(transform.position, targetPosition) > innerRadius)
            {
                // 基準の範囲内の場合ターゲットとする
                totalPosition += targetPosition - transform.position;
            }
        }

        return totalPosition.normalized;
    }

    /// <summary>
    /// エージェント以外のオブジェクトから距離をとるメソッド
    /// </summary>
    /// <returns>回避ベクトル</returns>
    public override Vector3 Avoid()
    {
        Vector3 vector = new(0, 0, 0);                  // 回避ベクトルの初期化
        Collider myCollider = GetComponent<Collider>();
        int innerObjectNum = 0;                         // 回避対象の数

        foreach (var obstacle in objectList)
        {
            Collider targetCollider = obstacle.GetComponent<Collider>();

            // 登録されている障害物リストを走査
            if (myCollider != null && targetCollider != null)
            {
                // 自身の位置に最も近いターゲットのコライダーの表面上の点を取得
                Vector3 closestPointOnTarget = targetCollider.ClosestPoint(transform.position);
                var distance = Vector3.Distance(transform.position, closestPointOnTarget);

                // innerRadius以内に障害物がある場合、回避ベクトルを加算
                if (distance <= innerRadius)
                {
                    vector += (transform.position - closestPointOnTarget).normalized * innerRadius / distance;
                    innerObjectNum++;
                }
            }
        }

        // 回避対象が1つ以上あれば平均化
        if (innerObjectNum > 0)
        {
            vector /= innerObjectNum;
        }

        return vector;
    }

    /// <summary>
    /// リーダーのメインミッションを実行するメソッド
    ///
    /// ここでは目的地へ進むミッションを返す
    /// </summary>
    /// <returns>ミッションのための移動ベクトル</returns>
    public override Vector3 ExecutedMission()
    {
        // return ExecuteForwardMission();
        return ExecuteTargetMission();
    }

    /// <summary>
    /// 前方へ進むミッションメソッド
    /// </summary>
    /// <returns>移動ベクトル</returns>
    public Vector3 ExecuteForwardMission()
    {
        Vector3 vector = transform.forward;
        vector.y = 0;
        vector = vector.normalized * alignPower;
        vector += Avoid() * separatePower;
        return vector.normalized;
    }

    /// <summary>
    /// 目的地へ進むミッションメソッド
    /// </summary>
    /// <returns>移動ベクトル</returns>
    public Vector3 ExecuteTargetMission()
    {
        Vector3 direction = (destination - transform.position).normalized;
        Vector3 vector = direction * alignPower + Avoid() * separatePower;

        // 目的地に十分近づいたら停止
        if (Vector3.Distance(transform.position, destination) < innerRadius)
        {
            vector = new Vector3(0, 0, 0);
        }

        return vector.normalized;
    }

}
