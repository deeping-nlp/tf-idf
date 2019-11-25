from TfIdf import TfIdf


class TestTfIdf:
    def test_similarity(self):
        table = TfIdf()
        # 训练语料：三篇文章
        table.add_document("doc1", ["The", "game", "of", "life", "is", "a", "game", "of", "everlasting", "learning"])
        table.add_document("doc2", ["The", "unexamined", "life", "is", "not", "worth", "living"])
        table.add_document("doc3", ["Never", "stop", "learning"])

        table.calculate_tf()
        table.calculate_idf()
        table.calculate_tf_idf()



        sims = table.similarities(["life","learning"])
        return sims


if __name__ == "__main__":
    testTfIdf_similarity = TestTfIdf().test_similarity()
    print(testTfIdf_similarity)