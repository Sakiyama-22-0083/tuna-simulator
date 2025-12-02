# Multi-AUV Aquaculture Simulator

近畿大学情報学部フィールド分散知能研究室で開発しているAUV協調制御によるマグロ養殖生簀のモニタリングのシミュレータ（ver1.1.0 以前はseaSimulatorというプロジェクト）

## フォルダ構成

```markdown
multi-auv-aquaculture-sim/
├── Assets/                     # Unityプロジェクトのアセット
│   ├── Scenes/                 # シーンファイル
│   ├── Scripts/                # スクリプト
│   ├── Prefabs/                # プレハブ
│   ├── Materials/              # マテリアル
│   ├── ThirdParty/             # 外部アセット
│   ├── Terrain/
│   ├── ProBuilder Data/
│   └──  ...
├── ProjectSettings/
├── Packages/
├── .gitignore
├── README.md
└── ...
```

## 各アセットについて

importした外部アセットはAssets/ThirdParty内にある．

### #NVJOB Boids

boidsの無料アセット
[Simple Boids - Flocks of birds, fish and insects (Unity Asset Store)](https://assetstore.unity.com/packages/3d/characters/animals/simple-boids-flocks-of-birds-fish-and-insects-164188?locale=ja-JP&srsltid=AfmBOooScCGx4V2nc0ns9el10fd2UzdcFvqkutNB3KIunNBPzSlfBORb)

### ABKaspo Games Assets

水のテクスチャ
[A.U.R.W HDRP Free Version](https://assetstore.unity.com/packages/vfx/shaders/a-u-r-w-hdrp-free-version-192148?srsltid=AfmBOop_IFtZlj-Xj7-uq5qvoHs4SCxAlBkQzh2yNtzZel159tzf_APO)

### Bitgem

水のテクスチャ
[URP Stylized Water Shader (Proto Series)](https://assetstore.unity.com/packages/vfx/shaders/urp-stylized-water-shader-proto-series-187485)

### Boat船

船のオブジェクト
[Boats PolyPack](https://assetstore.unity.com/packages/3d/vehicles/sea/boats-polypack-189866#content)

### Handpainted_Grass_and_Ground_Textures

地形のテクスチャ
[Handpainted Grass and Ground Textures](https://assetstore.unity.com/packages/2d/textures-materials/nature/handpainted-grass-ground-textures-187634#content)

### POLY_Submarine

AUVのオブジェクト
[POLY_Submarine](https://assetstore.unity.com/packages/3d/vehicles/sea/poly-submarine-232763#content)

### Water saface

水のテクスチャ
[Simple Water Shader URP](https://assetstore.unity.com/packages/2d/textures-materials/water/simple-water-shader-urp-191449#content)

### import元が不明なアセット

- Ground textures pack
- Stylized Ocean Materials
- Tree
- UnityTechnologies
- 沼地マテリアル

## 各Prefabについて

### net-01

マグロの生簀のパーツ
作成者: 松本

### マグロ

キハダマグロのフリー素材
[https://booth.pm/ja/items/1107229](https://booth.pm/ja/items/1107229)

### 作成者が不明なPrefab

- iphone 13 pro max
- タブレット

## Scripts

```markdown
multi-auv-aquaculture-sim/Assets/Scripts
├── CameraSwitcher.cs   # カメラを切り替える機能
├── fog.cs              # 水中を表現するための霧の色を自動更新する機能
├── mainScript.cs       # RGBパラメータフィッティングのメインスクリプト
├── particle1.cs        # 塵パーティクルを表示する機能
├── particles2.cs       # 泡パーティクルを表示する機能
├── watercamera.cs      # 水中でのみオブジェクトを表示する機能
└── parameter-01.csv    # パラーメータフィッティングした水のRGB値
```

## 開発者

- seaSimulator: 松本拓也，北里光希(それぞれの成果物をver1.0.0, ver1.1.0とする)
- ver1.1.1以降: 﨑山楓麻
