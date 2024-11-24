from datasets import load_dataset
from qdrant_client import models as qmodels
from qdrant_client import QdrantClient

ds = load_dataset("arbml/Hadith")
documents = ds['train']['Text']

from pylate import indexes, models, retrieve

model = models.ColBERT(
    model_name_or_path="jinaai/jina-colbert-v2",trust_remote_code=True
)
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
    override=True,
)

retriever = retrieve.ColBERT(index=index)
documents_ids = [str(i) for i in range(len(documents))]
documents_embeddings = model.encode(
    documents,
    batch_size=128,
    is_query=False,
    show_progress_bar=True

)

index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
    batch_size=4000
)

h_1_t = "من ترك صلاة مكتوبة متعمدا برئت منه ذمة الله"  # 40165
h_2_t = "من سن في الإسلام سنة حسنة"  # 37492
h_3_t = "الراحمون يرحمهم الرحمن"  # 25311
h_4_t = "مثل المؤنين في توادهم وتراحمهم كمثل الجسد الواحد"  # 36753
h_5_t = "ألا إن في الجسد مضغة إذا صلحت صلح الجسد كله"  # 81731
h_6_t = "إن العبد ليتكلم بالكلمة من سخط الله"  # 27672
h_7_t = "من تشبه بقوم فهو منهم"  # 23973
h_8_t = "اجتنبوا أم الخبائث"  # 5573
h_9_t = "ليس المؤمن باللعان و لا الطعان ولا الفاحش"  # 22751
h_10_t = "فضل العالم علي العابد كفضل القمر"  # 39828

h_1 = "العهد الذي  بيننا وبينهم الصلاة فمن تركها فقد كفر"  # 78826
h_2 = "من دعا الي الهدي كان له من الأجور مثل أجور متبعه"  # 27900
h_3 = "من لا يرحم الناس لا يرحمه الله"  # 30040
h_4 = "المؤمن للمؤمن كالبنيان يشد بعضه بعضا"  # 37904
h_5 = "إن الله لا ينظر الي صوركم و لكن ينظرالي قلوبكم"  # 29642
h_6 = "من كان يؤمن بالله واليوم الأخر فليقل خيرا أو ليصمت"  # 25437
h_7 = "يحشر المرء مع من يحب"  #
h_8 = "لعن الله الخمر وشاربه وساقيها"  # 24563
h_9 = "سباب المسلم فسوق وقتاله كفر"  # 4037
h_10 = "من سلك طريقا يلتمس فيه علما"  # 27070

queries_embeddings = model.encode(
    [h_1,h_2,h_3,h_4,h_5,h_6,h_7,h_8,h_9,h_10],
    batch_size=2,
    is_query=True,
    show_progress_bar=True
)


scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=10,
    batch_size=32
)

for score in scores:
    for el in scores:
        print(documents[int(el['id'])])