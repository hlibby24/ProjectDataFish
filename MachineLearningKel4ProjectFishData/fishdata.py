import streamlit as st
import joblib
import numpy as np

model = joblib.load('ML_FishData.pkl')
encoder = joblib.load('encoder2.pkl')

st.title("Fish Data")


Species_Name = st.selectbox('Nama Ikan', options=['Salmon', 'Tuna', 'Cod'])
Region = st.selectbox('Negara', options=['North Atlantic', 'Pacific Ocean', 'Mediterranean Sea'])
Breeding_Season = st.selectbox('Musim', options=['Summer', 'Auntum', 'Winter'])
Fishing_Method = st.selectbox('Pengambilan', options=['Jaring', 'Pancing', 'Bom'])
Fish_Population = st.selectbox('Populasi Ikan', options=['Rendah', 'Sedang', 'Tinggi'])
Average_Size_cm = st.selectbox('Rata-rata ukuran ikan', options=['Kecil', 'Sedang', 'Besar'])
Water_Temperature_C = st.selectbox('Suhu Air', options=['Cold', 'Warm', 'Hot'])
Water_Pollution_Level = st.selectbox('Level Polusi Air', options=['Low', 'Medium', 'High'])

if st.button("Prediksi"):
    try:
        all_labels = ['Salmon', 'Tuna', 'Cod',
                      'North Atlantic', 'Pacific Ocean', 'Mediterranean Sea',
                      'Summer', 'Auntum', 'Winter',
                      'Jaring', 'Pancing', 'Bom',
                      'Rendah', 'Sedang', 'Tinggi',
                      'Kecil', 'Sedang', 'Besar',
                      'Cold', 'Warm', 'Hot',
                      'Low', 'Medium', 'High']
        encoder.fit(all_labels)

        species_encoded = encoder.transform([Species_Name])[0]
        region_encoded = encoder.transform([Region])[0]
        breeding_season_encoded = encoder.transform([Breeding_Season])[0]
        fishing_method_encoded = encoder.transform([Fishing_Method])[0]
        fish_population_encoded = encoder.transform([Fish_Population])[0]
        average_size_encoded = encoder.transform([Average_Size_cm])[0]
        water_temperature_encoded = encoder.transform([Water_Temperature_C])[0]
        water_pollution_level_encoded = encoder.transform([Water_Pollution_Level])[0]

        data = np.array([[species_encoded, region_encoded, breeding_season_encoded, fishing_method_encoded, 
                          fish_population_encoded, average_size_encoded, water_temperature_encoded, 
                          water_pollution_level_encoded]])

        submit_label = model.predict(data)[0]
        
        Species_Name = encoder.inverse_transform([submit_label])[0]

        overfishing_risk = False
        
        if Fish_Population == 'Rendah' or (Fishing_Method == 'Bom' and Water_Pollution_Level == 'High'):
            overfishing_risk = True
        
        if overfishing_risk:
            st.warning("Warning: situasi ini merupakan indikasi overfishing")
        else:
            st.success("Populasi ikan masih aman")

    except ValueError as ve:
        st.error(f"ValueError: {ve}. Please check the input values.")
    except Exception as e:
        st.error(f"Error: {e}")