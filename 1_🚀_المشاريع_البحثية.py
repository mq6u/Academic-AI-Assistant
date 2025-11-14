import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- إعدادات الصفحة والاتصال ---
st.set_page_config(page_title="منشئ الأبحاث", page_icon="🚀")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ عذراً، خطأ في الاتصال بالخادم.")
    st.stop()

DB_DIR = "persistent_db" 

# --- تحميل الخزينة العلمية ---
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists(DB_DIR):
        return None 
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    return vector_store

vector_store = load_knowledge_base()

# --- واجهة المستخدم ---
st.title("🚀 منشئ المشاريع البحثية")
st.caption("يعمل هذا النظام بناءً على الخزينة العلمية (كتبك ومراجعك).")

if vector_store:
    st.subheader("1. أدخل متطلبات البحث أو الواجب:")
    requirements = st.text_area("الصق متطلبات البحث هنا:", height=200, placeholder="مثال: اكتب بحثاً من 5 صفحات عن...")
    
    if "research_paper" not in st.session_state:
        st.session_state.research_paper = ""

    if st.button("🚀 ابدأ بإنشاء البحث", type="primary"):
        if not requirements:
            st.warning("الرجاء إدخال المتطلبات أولاً.")
        else:
            with st.spinner("جارٍ تحليل المتطلبات والبحث في الخزينة العلمية... (قد يستغرق هذا عدة دقائق)..."):
                try:
                    docs = vector_store.similarity_search(requirements, k=25)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=api_key,
                        temperature=0.5
                    )
                    
                    prompt = f"""
                    أنت بروفيسور وخبير في كتابة الأبحاث الأكاديمية.
                    
                    [متطلبات الطالب]:
                    {requirements}
                    
                    [المعلومات المسترجعة من الخزينة العلمية]:
                    {context_text}
                    
                    تعليمات صارمة (نسخة v2 - ضد النسخ):
                    1. قم بإنشاء بحث أكاديمي "كامل" ومتماسك.
                    2. يجب أن تلتزم "بالكامل" بالمتطلبات المذكورة.
                    3. استخدم "حصرياً" المعلومات المسترجعة من الخزينة.
                    4. [الأمر الأهم]: يجب عليك إعادة صياغة كل المعلومات بأسلوبك الأكاديمي الخاص. ممنوع منعاً باتاً نسخ أي جملة حرفياً (No direct quotes) من المعلومات المسترجعة.
                    5. يجب أن يكون الناتج النهائي فريداً 100% في الصياغة.
                    6. لا تذكر "المصدر" أو "المعلومات"، بل أجب بثقة.
                    """
                    
                    response = llm.invoke(prompt)
                    
                    st.session_state.research_paper = response.content
                    
                    st.markdown("### 📝 مسودة البحث الأولية:")
                    st.success("تم إنشاء البحث بنجاح! راجع المسودة أدناه.")
                    st.markdown(st.session_state.research_paper)
                            
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التوليد: {e}")

    # زر التحميل
    if st.session_state.research_paper:
        st.divider()
        st.subheader("2. تحميل البحث")
        st.download_button(
            label="📥 تحميل البحث كملف (.txt)",
            data=st.session_state.research_paper,
            file_name="MyResearchPaper.txt",
            mime="text/plain"
        )

else:
    st.error("⚠️ لم يتم العثور على الخزينة العلمية. يرجى تشغيل `build_database.py` أولاً.")

# --- التذييل المحدث ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #808080; font-size: 14px;'>
    الذكاء الاصطناعي لخدمة البحث العلمي 📚 | 2025
    </div>
    """, 
    unsafe_allow_html=True
)