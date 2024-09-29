# LangChainGraphRAG

## 为什么想做这样一个仓库？
主要原因是实习的时候mentor喊我搞GraphRAG，但是微软官方开源的库实在是一言难尽。里面夹杂各种私货就算了，代码耦合度也非常高。
因此，参考了各种各样的几十篇博客以及各种具体代码实现后，决定采用LangChain+Neo4j+GraphRAG的实现。
主要参考了如下几个开源的工作。
1. [微软官方](https://github.com/microsoft/graphrag)
2. [Tomaz Bratanic老哥写的Index构建过程](https://github.com/tomasonjo/blogs/blob/master/llm/ms_graphrag.ipynb)
3. [ollama+graphrag](https://github.com/TheAiSingularity/graphrag-local-ollama)
4. [Kapil Sachdeva老哥写的Query构建过程](https://github.com/ksachdeva/langchain-graphrag/tree/main)
其实说是自己的工作也不太准确，大概就是个超级大杂烩，里面有自己的东西也有很多别人已经做完的工作，特别是Leiden构建这块，如果不是Tomaz Bratanic老哥
写的查询语句，靠自己估计很难想到。但是也算是集百家之长吧。
- 微软提供了思路以及具体实现步骤，但是泛化性很差，<span style="color:blue">因此考虑使用LangChain实现</span>
- Tomaz Bratantic提供了Index构建过程，但是目前还没有放出来Query如何写，并且Index中仍然缺失相当一部分信息，比如text_units_ids等。
- ollama+graphrag给我提供了一个未来的方向-->究竟什么样的方法能让graphrag普适大众？非api的本地化模型。但是诸多缺陷仍然存在<span style="color:green">，我也是还未完成这部分的工作</span>。
特别是中间各个步骤中，7b，13b模型指令没法跟随该怎么办；或许力大砖飞真的在graphrag这个项目中体现的淋漓尽致。
- Kapil Sachdeva老哥的query模块写的非常好，每个模块都独立性非常高，我也大都拿来用了。唯一可惜的是里面没有支持Neo4j的实现，而是用类似于graphml这样的图表示来实现。或许也是一个
方向吧，但是享受着neo4j的便利的我还是想着用neo4j来替代。
- 此外，增加了一些特性，包括考虑到Community Neo4j不支持多database，所以给所有标签加了user_id来解决多用户问题等

## 其他
刚刚入手GraphRAG一个月，甚至可以说刚刚接触知识图谱一个月，感觉里面的坑多的可以发很多paper。且行且走，道阻且长~

## TODO
- [ ] 提供代码
- [ ] pip requirements
- [ ] neo4j环境配置
- [ ] ollma集成
- [ ] quick start