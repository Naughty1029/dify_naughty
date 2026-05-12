# 標準ライブラリ
import argparse
import shutil

# LangChain core
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# LangChain その他モジュール
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings


# 利用モデルとDB定義
model = OllamaLLM(model="gemma3:1b", base_url="http://localhost:11434")
emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
db = Chroma(
    collection_name="langchain",
    embedding_function=emb,
    persist_directory="./chroma_db",
)


# ローカルディレクトリの読み込み対象ファイルを読込み、インデキシング。
def create_index():
    print("ドキュメントを読み込んでDBに登録します。")
    ldr = DirectoryLoader(
        "./local_documents",  # 登録するドキュメントを配置するローカルディレクトリパス
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    # ドキュメントの読み込みを実施
    raw_docs = ldr.load()

    # 読み込んだドキュメントをチャンクに分割。
    txt_sp = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=25,
        separators=["\n\n", "。"],
    )
    docs = txt_sp.split_documents(raw_docs)

    # ロードしたドキュメントのDB登録(indexing)
    db.add_documents(documents=docs)

    print("インデキシングが完了しました。")


# DBのクリア
def delete_index():
    print("データベースをクリアします。")
    db.delete_collection()
    shutil.rmtree("./chroma_db")
    print("データベースのクリアが完了しました。")


# LLMに関する質問処理
def query_llm(user_pmt: str):
    print(f"質問を受け付けました: {user_pmt}")

    # DBから質問に関連するドキュメントを得るIF(リトリーバ)を作成
    retriever = db.as_retriever(search_kwargs={"k": 1})
    context_docs = retriever.invoke(user_pmt)

    if len(context_docs) == 0:
        print("関連する情報が見つかりません。情報未登録の可能性があります。")
        return

    # LangChainのプロンプトテンプレート
    pmt_all = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: """
{user_pmt}
"""
''')

    # LangChainのチェイン定義
    chain = (
        {"context": retriever, "user_pmt": RunnablePassthrough()}
        | pmt_all
        | model
        | StrOutputParser()
    )

    # ユーザの入力をネタにchainを実行
    ai_msg = chain.invoke(user_pmt)
    print(f"LLMの応答: {ai_msg}")


# コマンドライン引数の解析
def main():
    parser = argparse.ArgumentParser(description="LangChainベースのRAGシステム")
    parser.add_argument(
        "-a", "--add", action="store_true", help="ドキュメント登録・インデックス作成"
    )
    parser.add_argument(
        "-d", "--delete", action="store_true", help="データベースクリア"
    )
    parser.add_argument("-q", "--query", type=str, help="LLMへ質問文を行う")
    args = parser.parse_args()

    if args.add:
        create_index()
    elif args.delete:
        delete_index()
    elif args.query:
        query_llm(args.query)
    else:
        print("いずれかのオプションを指定してください。-h オプションで使用方法を確認できます。")


if __name__ == "__main__":
    main()