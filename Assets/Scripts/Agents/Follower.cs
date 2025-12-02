using UnityEngine;

/// <summary>
/// フォロワークラス
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class Follower : Boid
{
    private GameObject leader;
    public void SetLeader(GameObject leader)
    {
        this.leader = leader;
    }

    /// <summary>
    /// フォロワーのメインミッションを実行するメソッド
    ///
    /// ここでは目的地へ進むミッションを返す
    /// </summary>
    /// <returns>ミッションのための移動ベクトル</returns>
    public override Vector3 ExecutedMission()
    {
        var targetPosition = leader.transform.position;
        Vector3 direction = (targetPosition - transform.position).normalized;
        Vector3 vector = direction;

        vector = Separation() * separatePower + Align() * alignPower + Cohesion() * cohesionPower;

        return vector.normalized;
    }

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
        Rigidbody leaderRB = leader.GetComponent<Rigidbody>();
        vector += leaderRB.velocity;
        return vector.normalized;
    }

    /// <summary>
    /// リーダーの中心座標を目指すメソッド
    /// </summary>
    /// <returns>結合ベクトル</returns>
    public override Vector3 Cohesion()
    {
        Vector3 vector = new(0, 0, 0);
        var targetPosition = leader.gameObject.transform.position;

        if (Vector3.Distance(transform.position, targetPosition) > innerRadius)
        {
            vector += targetPosition - transform.position;
        }

        return vector.normalized;
    }

    /// <summary>
    /// エージェント以外のオブジェクトから距離をとるメソッド
    /// </summary>
    /// <returns>回避ベクトル</returns>
    public override Vector3 Avoid()
    {
        Vector3 vector = new(0, 0, 0);
        Collider myCollider = GetComponent<Collider>();
        int innerAgentNum = 0;

        foreach (var obstacle in objectList)
        {
            Collider targetCollider = obstacle.GetComponent<Collider>();

            if (myCollider != null && targetCollider != null)
            {
                // 自身の位置に最も近いターゲットのコライダーの表面上の点を取得
                Vector3 closestPointOnTarget = targetCollider.ClosestPoint(transform.position);
                var distance = Vector3.Distance(transform.position, closestPointOnTarget);

                if (distance <= innerRadius)
                {
                    vector += (transform.position - closestPointOnTarget).normalized * innerRadius / distance;
                    innerAgentNum++;
                }
            }
        }

        // 回避対象が1つ以上あれば平均化
        if (innerAgentNum > 0)
        {
            vector /= innerAgentNum;
        }

        return vector;
    }

}
