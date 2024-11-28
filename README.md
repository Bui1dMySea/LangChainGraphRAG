<h1 align="center">欢迎来到LangChainGraphRAG项目 👋</h1>

> Building Intelligent Applications: The Powerful Combination of LangChain, Neo4j, and GraphRAG

### 🏠 [主页](https://github.com/Bui1dMySea/LangChainGraphRAG)

## 📌 前言：为什么想做这样一个项目？
主要原因是微软官方开源的库实在是一言难尽。里面夹杂各种私货就算了，代码耦合度也非常高。
因此，参考了各种各样的几十篇博客以及各种具体代码实现后，决定采用LangChain+Neo4j+GraphRAG的实现。
目前，主要参考了如下几个开源的工作。

- [微软官方](https://github.com/microsoft/graphrag)
- [Tomaz Bratanic老哥写的Index构建过程](https://github.com/tomasonjo/blogs/blob/master/llm/ms_graphrag.ipynb)
- [ollama+graphrag](https://github.com/TheAiSingularity/graphrag-local-ollama)
- [Kapil Sachdeva老哥写的Query构建过程](https://github.com/ksachdeva/langchain-graphrag/tree/main)

## 🚀 快速开始

### Conda Environment

```sh
conda create -n langchain-graphrag python=3.10
conda activate langchain-graphrag
pip install -r requirements.txt
```

### Neo4j Install

1. 获取访问通道

```Bash
mkdir /etc/apt/keyrings # 可选
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg
echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
```

2. 显示结果

```Bash
apt list -a neo4j
```

3. 安装neo4j版本

```Bash
sudo apt-get install neo4j=1:5.21.0
```

4. 修改配置文件权限+修改配置文件

```Bash
chmod +x /etc/neo4j/neo4j.conf # 这里是Debian路径；其余的可以访问neo4j官网查看默认路径
vim /etc/neo4j/neo4j.conf

# 找到或者直接修改取消注释掉这两行
dbms.security.procedures.unrestricted=gds.*,apoc.*
dbms.security.procedures.allowlist=apoc.coll.*,apoc.load.*,gds.*,apoc.*
```

5. 下载两个文件

   a.  `apoc-{你的版本}-core.jar` https://github.com/neo4j/apoc/releases/ # 仅限于高于4.4.x的，低于4.4.x的版本需要自己去找 

   b.  `neo4j-graph-data-science-{你的版本}.jar` https://github.com/neo4j/graph-data-science/releases/ # https://neo4j.com/docs/graph-data-science/current/installation/supported-neo4j-versions/ 这里是neo4j版本与datascience的具体对照表

6. 将刚刚下好的两个文件复制到*/var/lib/neo4j/plugins*路径下 # Debian路径;其余系统的路径访问neo4j官网查看

7. 启动neo4j: `sudo neo4j start` ｜ 注意！！如果已经在步骤 6 之前已经启动了neo4j，需要重启来应用配置文件 `sudo neo4j restart`

### 配置api-key

`export OPEN_API_KEY=YOUR_API_KEY` 或者运行build_index.py以及search.py文件时等待需要脚本要求输入api_key。

### 构建索引

` bash index.sh`

### 查询

```bash
python search.py \
--neo4j_uri bolt://localhost:7687 \
--neo4j_username your_username \
--neo4j_password  your_password \
--model_provider openai \
--embedding_model_name_or_path BAAI/bge-m3 \
--chat_model_name deepseek-chat \
--base_url https://api.deepseek.com \
--completion_mode [completion,chat] \
--query_mode [local,global]
```

### 更多方案(ollama,huggingface)
请参考[更多Examples](./example/)

## 👦🏻 作者

 **Weijie Liu**

* 主页: https://github.com/Bui1dMySea
* Github: [@Bui1dMySea](https://github.com/Bui1dMySea)

## 🤝 贡献

欢迎各位大佬给issue、fork、pull requests等<br />如果有问题也请提问 [issues page](https://github.com/Bui1dMySea/LangChainGraphRAG/issues). 

## ⭐️ 喜欢的请点个免费的star~

走过路过不要错过！留下一个免费的赞吧~球球了⭐️
