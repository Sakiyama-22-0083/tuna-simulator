using UnityEngine;

/// <summary>
/// Leaderと複数のFollowerオブジェクトを指定した範囲内のランダムな位置に生成し，
/// 各フォロワーにリーダーへの参照を持たせるためのクラス
/// </summary>
public class LFGenerator : MonoBehaviour
{
    public GameObject leader;       // 生成するリーダーのプレハブ
    public GameObject follower;     // 生成するフォロワーのプレハブ
    public int followerNum = 5;     // フォロワーの数
    public float spawnRadius = 10f; // 生成する座標範囲

    void Start()
    {
        SpawnLeader(leader);
        SpawnFollower(follower, followerNum);
    }

    /// <summary>
    /// 指定範囲内のランダムな位置にリーダーを1体生成するメソッド
    /// </summary>
    /// <param name="leader">生成するリーダーのプレハブ</param>
    void SpawnLeader(GameObject leader)
    {
        Vector3 randomOffset = Random.insideUnitSphere * spawnRadius;
        // y座標を正にする
        randomOffset.y = Mathf.Abs(randomOffset.y);
        Vector3 spawnPosition = transform.position + randomOffset;

        this.leader = Instantiate(leader, spawnPosition, Quaternion.identity);
    }

    /// <summary>
    /// 指定範囲内のランダムな位置に複数のフォロワーを生成しµ
    /// 各フォロワーにリーダーへの参照を渡すメソッド
    /// </summary>
    /// <param name="follower">生成するフォロワーのプレハブ</param>
    /// <param name="followerNum">生成するフォロワーの数</param>
    void SpawnFollower(GameObject follower, int followerNum)
    {
        for (int i = 0; i < followerNum; i++)
        {
            Vector3 randomOffset = Random.insideUnitSphere * spawnRadius;
            // y座標を正にする
            randomOffset.y = Mathf.Abs(randomOffset.y);
            Vector3 spawnPosition = transform.position + randomOffset;

            GameObject followerInstance = Instantiate(follower, spawnPosition, Quaternion.identity);
            Follower followerScript = followerInstance.GetComponent<Follower>();
            followerScript.SetLeader(this.leader);
        }
    }

}
