# 协作规范

## 分支策略

```
main          ← 稳定发布版本，只从 develop 合并
  └── develop ← 开发主线，所有功能分支从此拉出
        ├── feature/source-coding     ← 成员 A
        ├── feature/channel-coding    ← 成员 B
        ├── feature/channel-model     ← 成员 C
        └── feature/evaluation        ← 成员 D
```

| 分支 | 用途 | 说明 |
|---|---|---|
| `main` | 稳定版本 | 禁止直接提交，仅从 `develop` 合并 |
| `develop` | 开发主线 | 各功能分支的合入目标 |
| `feature/*` | 功能开发 | 从 `develop` 拉出，完成后合回 `develop` |

## 开发流程

### 1. 从 develop 拉出功能分支

```bash
git checkout develop
git pull origin develop
git checkout -b feature/<模块名>
```

### 2. 在功能分支上开发

保持提交粒度合理，每个小任务一个提交。提交信息使用中文，格式：

```
<动词>: <简要描述>

- 做了 A
- 做了 B
```

常用动词：
- `新增` — 新文件或新功能
- `修改` — 改动已有代码
- `修复` — bug 修复
- `重构` — 不改变行为的代码整理

例如：
```
新增: DCT 变换编码器

- 实现 8×8 分块和 2D-DCT 变换
- 添加 JPEG 标准量化表
```

### 3. 合并回 develop

功能完成后（自测通过），发起 Pull Request 或本地合并：

```bash
git checkout develop
git pull origin develop
git merge feature/<模块名>
git push origin develop
```

### 4. 同步上游变更

如果 develop 有其他人合入了新代码，自己的功能分支需要同步：

```bash
git checkout feature/<模块名>
git merge develop
```

## 合并到 main

- 所有功能模块集成测试通过后，由仓库管理员（成员 C）将 `develop` 合并到 `main`
- `main` 作为可运行、可提交的稳定快照

## 目录约定

- 新模块代码放入 `src/<模块名>/`，不要修改他人目录
- 数据文件放在 `data/`
- 实验结果输出到 `results/`
- 临时文件放在 `output/`，不需要提交

## 注意事项

- **禁止直接在 develop 上提交代码**，必须走功能分支
- **禁止直接向 main 提交代码**，必须通过 develop 合并
- 提交前确保代码能正常运行，不破坏其他人的模块
- 不要提交 `__pycache__`、`.pyc`、虚拟环境等文件（.gitignore 已配置）
