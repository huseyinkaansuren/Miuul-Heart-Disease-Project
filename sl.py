import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# For loading your trained model
from sklearn.preprocessing import StandardScaler
from helper import preprocess_data, grab_col_names

st.set_page_config(layout="wide")
st.markdown("""<h1 style='color: #3498db; text-align: center;'>️👩‍⚕️Kalp Hastalığı Tahmini👨‍⚕️</h1>""",unsafe_allow_html=True)

#HOME
tab_home, tab_data, tab_vis, tab_model = st.tabs(["Ana Sayfa", "Veri Seti", "Veriye Genel Bakış", "Model"])



tab_home.subheader("Biz Kimiz?")
tab_home.write("Merhaba! Biz, MİNİ DATATAM EĞİTİM VE ARAŞTIRMA HASTANESİ, sağlık sektöründe bir dizi hizmet sunan bir ekip olarak sizlere hizmet vermeye başladık. Tesisimize gelen hastaların durumlarını inceleyerek, kalp hastalığı olup olmadığını tahmin etmek üzere geliştirdiğimiz bir uygulama ile sağlığınıza odaklanıyoruz.")


tab_home.subheader("Kalp Hastalığı Tahminleme Uygulaması")
tab_home.write("MİNİ DATATAM EĞİTİM VE ARAŞTIRMA HASTANESİ olarak, geliştirdiğimiz kalp hastalığı tahminleme uygulaması ile hastalarımıza daha iyi hizmet sunmaya odaklanıyoruz. Bu uygulama, tesisimize gelen hastaların sağlık durumlarını inceleyerek, kalp hastalığı riskini tahminlememize yardımcı oluyor.")
#tab_home.image("media/heartimage.jpg")

tab_home.subheader("Hedeflerimiz📝")
hedefler = [
    ':red[Doğru Tahminler]: Uygulamamız, hastaların sağlık verilerini kullanarak kalp hastalığı olup olmadığını doğru bir şekilde tahminlemeyi hedefler.',
    ':red[Erken Teşhis]: Hastaların sağlık durumlarını hızlı ve etkili bir şekilde analiz ederek erken teşhis konulmasına katkı sağlamayı amaçlarız.',
    ':red[Hasta Bilincini Arttırma]: Toplumda sağlığa dair farkındalığı arttırarak hastaların kendi sağlıklarına daha fazla dikkat etmelerine destek olmayı hedefleriz.'
]

for hedef in hedefler:
    tab_home.markdown(f'- {hedef}', unsafe_allow_html=True)




#DATA VIS.
df = pd.read_csv("new_train30k.csv")
tab_data.subheader("Veri Seti")
tab_data.write("Hastanelerle olan anlaştmamızdan ve bizimle farklı yollardan irtibata geçen insanlardan elde ettiğimiz bilgileri kullanarak oluşturulmuş bir veri setidir.")


data = {
    'Değişken İsimleri': ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
                          'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                          'NoDocbcCost', 'GenHlth', 'Menthlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 'HeartDiseaseorAttack'],
    'Açıklamalar': ['Kişiye bir sağlık profesyoneli tarafından Yüksek Kan Basıncı (High Blood Pressure) teşhisi konup konulmadığını belirtir.',
                    'Kişiye bir sağlık profesyoneli tarafından Yüksek Kan Kolesterolü (High Blood Cholesterol) teşhisi konup konulmadığını belirtir.',
                    'Kişinin son 5 yıl içinde kolesterol seviyelerinin kontrol edilip edilmediğini belirtir.',
                    'Vücut Kitle İndeksi (BMI), kişinin kilosunu (kilogram cinsinden) boyunun karesine (metre cinsinden) bölerek hesaplanır.',
                    'Kişinin en az 100 sigara içip içmediğini belirtir.',
                    'Kişinin geçmişte felç geçirip geçirmediğini belirtir.',
                    'Kişinin şeker hastalığı geçmişi, şu anda prediyabetik olup olmadığı veya herhangi bir türde şeker hastalığına sahip olup olmadığını belirtir.',
                    'Kişinin günlük rutininde herhangi bir fiziksel aktivite olup olmadığını belirtir.',
                    'Kişinin günde 1 veya daha fazla meyve tükettiğini belirtir.',
                    'Kişinin günde 1 veya daha fazla sebze tükettiğini belirtir.',
                    'Kişinin haftada 14 ten fazla içki içip içmediğini belirtir.',
                    'Kişinin herhangi bir sağlık sigortasına sahip olup olmadığını belirtir.',
                    'Kişinin son 1 yıl içinde doktora gitmek istediği ancak maliyet nedeniyle gidemediğini belirtir.',
                    'Kişinin genel sağlığına verdiği yanıtı belirtir; 1 (mükemmel) ile 5 (zayıf) arasında değişir.',
                    'Kişinin son 30 günde kötü ruh sağlığı yaşadığı gün sayısını belirtir.',
                    'Kişinin son 30 günde kötü fiziksel sağlık yaşadığı gün sayısını belirtir.',
                    'Kişinin yürüme veya merdiven çıkarken zorlanıp zorlanmadığını belirtir.',
                    'Kişinin cinsiyetini belirtir; 0 kadın, 1 erkek.',
                    'Kişinin yaşını belirtir; 1, 18 ila 24 yaş arası, 13, 80 yaş ve üstü, her aralık 5 yıllık bir artışa sahiptir.',
                    'Kişinin tamamladığı en yüksek okul yılını belirtir; 0, hiç katılmamış veya sadece anaokulu, 6, 4 yıl veya daha fazla kolej okumuş. ',
                    'Kişinin toplam hane gelirini belirtir; 1 (en az 10.000 $) ile 8 (75.000 $ ve üzeri) arasında değişir.',
                    'Kişinin hastalığı olup olmaması durumu (0 yok, 1 var)']
}

tab_data.table(pd.DataFrame(data))

tab_vis.subheader("Veriye Genel Bakış")
tab_vis.write(df.head())





viscol_left, viscol_right = tab_vis.columns(2)
viscol_left.write(df.describe().T)

viscol_right.write(df.corr())

viscol_left.subheader("Cinsiyete Göre Hastalık Dağılımı")
viscol_left.image("media/cinshasgraf.jpeg")

viscol_right.subheader("Cinsiyete Göre Gelir Dağılımı")
viscol_right.image("media/cinsgelirgraf.jpeg")




#MODEL

# Load your trained model
model = joblib.load("final_model.joblib")  # Replace with the actual path to your model

modcol_left, modcol_right = tab_model.columns(2)

cols = df.columns[1:]

input_data = {}



input_data[cols[0]] = modcol_left.number_input("Yüksek Kan Basıncı teşhisiniz var mı? (Var ise 1, yok ise 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[1]] = modcol_right.number_input("Yüksek Kan Kolesterolü teşhisiniz var mı? (Var ise 1, yok ise 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[2]] = modcol_left.number_input("Son 5 yıl içerisinde kolesterol seviyeniz kontrol edildi mi? (Edildi ise 1, edilmedi ise 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[3]] = modcol_right.number_input("BMI değerinizi giriniz.", min_value = df["BMI"].min(), max_value = df["BMI"].max(), step=0.5)

input_data[cols[4]] = modcol_left.number_input("Sigara Kullanıyormusunuz? (Kullanıyor iseniz 1, kullanmıyor iseniz 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[5]] = modcol_right.number_input("Geçmişte felç geçirdiniz mi? (Geçirdiyseniz 1, geçirmediyseniz 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[6]] = modcol_left.number_input("Şeker hastalığı durumunuz nedir? (Hastalığınız yoksa 0, Prediyabetik iseniz 1, Şeker hastası iseniz 2)", min_value = 0, max_value = 2, step=1)

input_data[cols[7]] = modcol_right.number_input("Gün içinde fiziksel bir aktivite yapıyor musunuz? (Evet ise 1, değil ise 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[8]] = modcol_left.number_input("Gün içinde meyve tüketiyor musunuz? (Evet ise 1, değil ise 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[9]] = modcol_right.number_input("Gün içinde sebze tüketiyor musunuz? (Evet ise 1, değil ise 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[10]] = modcol_left.number_input("Bir hafta içerisinde 14' ten fazla içki içiyor musunuz? (Evet ise 1, değil ise 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[11]] = modcol_right.number_input("Sağlık sigortanız var mı? (Varsa 1, yoksa 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[12]] = modcol_left.number_input("Son 1 yıl içerisinde maliyet nedeniyle doktora gidemediğiniz oldu mu? (Gidemediyseniz 1, gittiyseniz 0)", min_value = 0, max_value = 1, step=1)

input_data[cols[13]] = modcol_right.number_input("1 ile 5 arasında genel sağlığınızı skorlayınız. (1 Mükemmel, 5 kötü)", min_value = 1, max_value = 5, step=1)

input_data[cols[14]] = modcol_left.number_input("Son 30 günde kötü ruh sağlığı yaşadığınız gün sayısı nedir?", min_value = 0, max_value = 30, step=1)

input_data[cols[15]] = modcol_right.number_input("Son 30 günde kötü fiziksel sağlık yaşadığınız gün sayısı nedir?", min_value = 0, max_value = 30, step=1)

input_data[cols[16]] = modcol_left.number_input("Merdiven çıkarken veya yürürken zorlanır mısınız? (Evet ise 1, değil ise 0)", min_value = 0, max_value = 1, step=1)



#Cinsiyet
gender = modcol_right.selectbox("Cinsiyetinizi seçiniz", options=["Kadın", "Erkek"])

input_data[cols[17]] = 0 if gender == "Kadın" else 1

#Yaş Aralığı

age_intervals = ["18-24", "24-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]

age_group = modcol_left.selectbox("Yaş aralığınıza göre girdiğiniz kategoriyi seçiniz.", options = age_intervals)

input_data[cols[18]] = age_intervals.index("18-24") + 1


#Eğitim düzeyi
education_intervals = ["Okula gitmedim veya sadece anaokuluna gittim", "İlkokul mezun", "Ortaokul mezun", "Lise mezun", "Önlisans", "Lisans ve üstü"]

education = modcol_right.selectbox("Eğitim düzeyinizi seçiniz.", options=education_intervals)

input_data[cols[19]] = education_intervals.index(education)


#Gelir aralığı
income_intervals = [
    "10,000 $ veya altı",
    "10,000 $ - 20,000 $",
    "20,000 $ - 30,000 $",
    "30,000 $ - 40,000 $",
    "40,000 $ - 50,000 $",
    "50,000 $ - 60,000 $",
    "60,000 $ - 75,000 $",
    "75,000 $ ve üzeri"
]

income = tab_model.selectbox("Gelir skalanızı giriniz.", options= income_intervals)

input_data[cols[20]] = income_intervals.index(income) + 1


#input_data

# Button to trigger prediction for a random user
if tab_model.button("Tahminle"):

    #tab_model.write(input_data)
    x = preprocess_data(input_data)
    #Kullanıcı olasılık
    # Use the preprocessed data for prediction
    probability = model.predict_proba(x)[:, 1][0]  # Probability of class 1
    tab_model.write(probability)

    # Display prediction result message
    if probability > 60:
        tab_model.subheader("Tahmin sonucu:")
        tab_model.write(
            f"%{probability * 100:.2f} olasılıkla hasta olarak tahminlendirildiniz. Kontrolünüz için başvuru sistemimizden en yakın tarihe randevu almanızı tavsiye ederiz."
        )
    else:
        tab_model.subheader("Tahmin sonucu:")
        tab_model.write("Bir kalp sorununuz bulunmamaktadır.")
        tab_model.balloons()