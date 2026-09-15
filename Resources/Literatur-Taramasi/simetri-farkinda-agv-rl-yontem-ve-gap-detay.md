# Ayrıntılı Gap & Yöntem Analizi: Simetri-Farkında AGV RL

Bu dosya, ana rapordaki (`simetri-farkinda-agv-rl-literatur-taramasi.md`) Bölüm 3 (Gap) ve Bölüm 4 (Yöntem Karşılaştırması) özetlerini derinleştirir.

---

## Gap 1: Eşdeğişken/simetri-artırımlı DRL, AGV/AMR sim-to-real transferiyle birleştirilmemiş

**Kanıt ve ayrıntı:**
- **Theile, Cao, Caccamo, Sangiovanni-Vincentelli (2024, IROS)** — "Equivariant Ensembles and Regularization for Reinforcement Learning in Map-based Path Planning". Bu, Fikir 5'e en yakın çalışma: harita-tabanlı yol planlamasında eşdeğişken politika/değer ağları için özel bileşen gerektirmeyen bir "eşdeğişken topluluk" (equivariant ensemble) yöntemi + eğitimde eşdeğişkenlik yönünde bir düzenlileştirme terimi öneriyorlar. Örnek verimliliği ve performansta kazanım gösteriyorlar. **Ancak:** çalışma tamamen simülasyon içinde kalıyor — hiçbir fiziksel robot deneyi veya sim-to-real transfer ölçümü yok. Bu, Fikir 5'in "sim-to-real simetri-kırılma açığı" katkısının doğrudan doldurabileceği boşluk.
- **Chang, Park, Seo, Horowitz, Lee, Choi (2025)** — "Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments" (PI-MDP). Bu çalışma tam olarak "simetri kırılması" sorununu ele alıyor ve Partially-Invariant MDP çerçevesi + PE-DQN/PE-SAC algoritmalarını öneriyor. Deneyler Grid-World, lokomosyon ve manipülasyon benchmark'larında yapılıyor. **Ancak:** navigasyon/AGV alanına hiç değinmiyor; sim-to-real deneyi yok (tamamen simülasyon-içi karşılaştırma).
- **Nguyen, Baisero, Klee, Wang, Platt, Amato (2024, CoRL)** — "Equivariant Reinforcement Learning under Partial Observability" gerçek donanımda test ediyor ("simulation and real hardware") ama görev robotik manipülasyon; AGV/mobil navigasyon değil, ve harita simetrisi değil nesne/görev simetrisi kullanılıyor.

**Mekanizma analizi (neden bu boşluk var):** Eşdeğişken-RL literatürü ağırlıklı olarak robot **kol/manipülasyon** camiasından (Platt, Walters ve öğrencileri — Northeastern/UMass ekseni) çıkmış; bu grup SO(2)/SO(3) eşdeğişkenliğini grasping/manipülasyon simetrilerine (nesne döndürüldüğünde grasp de dönmeli) uyguluyor, donanım da masaüstü robot kolu. AGV/AMR camiası ise ayrı bir literatür (endüstriyel mühendislik/lojistik kökenli, Bölüm G1'de listelenen kaynaklar) ve simetri kavramını hiç kullanmıyor. İki topluluk arasında **yöntem transferi olmamış** — bu bir "teknik zorluk" değil, camialar arası bilgi akışı eksikliği (disiplinlerarası boşluk).

**Somut araştırma yönü önerisi:** Theile ve ark. (2024)'ün eşdeğişken topluluk yöntemini AGV occupancy-grid navigasyonuna uyarlayıp, Peng ve ark. (2018) tarzı bir dinamik-randomizasyon sim-to-real hattına ekleyerek, iki koşulu (simetrik veri artırımlı PPO vs. eşdeğişken ağ) fiziksel bir AMR'da karşılaştırmak — tam olarak Fikir 5'in önerdiği tasarım.

**Yakın-ıskalayan çalışmalar:** Ordoñez Apraez ve ark. (2025) "Morphological symmetries in robotics" (IJRR) — robotun kendi gövde simetrisini (morphological symmetry) kullanıyor, ortam simetrisini değil; kavramsal olarak ilişkili ama ölçek farklı (proprioception vs. dış ortam haritası).

---

## Gap 2: AGV/depo RL literatüründe simetri açık bir tasarım parametresi olarak kullanılmıyor

**Kanıt ve ayrıntı:**
- **Oyekanlu ve ark. (2020, IEEE Access)** — AGV teknolojileri kapsamlı incelemesi; iletişim, kontrol teknolojileri üzerine odaklanıyor, RL veya simetri hiç geçmiyor.
- **Malus, Kozjek, Vrabič (2020, CIRP Annals)** ve **Kozjek, Malus, Vrabič (2021, Sensors)** — AMR filosu için MARL tabanlı sevkiyat/rota üretimi; ödül fonksiyonu ve ağ mimarisi tamamen simetriden bağımsız (standart MLP/tablo tabanlı Q-öğrenme).
- **Wesselhöft, Hinckeldeyn, Kreutzfeldt (2022, Robotics)** — AMR filoları için RL'in kısa bir taraması; makale RL algoritma seçimini (DQN, PPO, vb.) tartışıyor ama mimari-düzey inductive bias'lardan (simetri, eşdeğişkenlik) hiç bahsetmiyor.
- **Ellithy ve ark. (2024)** — AGV + Endüstri 4.0 çerçevesi, tamamen sistem-mühendisliği/entegrasyon odaklı.

**Mekanizma analizi:** Bu literatür, RL'i çoğunlukla bir **optimizasyon/çizelgeleme aracı** olarak görüyor (filo sevkiyatı, rota atama) — yani "hangi AMR hangi işe gitsin" sorusu, "AMR fiziksel olarak nasıl hareket etsin" sorusundan ayrışmış. Simetri kavramı düşük seviye kontrol/navigasyon problemine ait bir inductive bias; ama bu literatür yüksek seviye (task allocation/scheduling) problemine odaklanmış, düşük seviye kontrol genellikle klasik yöntemlere (A*, DWA) bırakılmış. Simetri-farkında RL literatürü ise düşük seviye kontrol/navigasyona odaklı ama akademik olarak AGV/depo uygulama bağlamından kopuk. **Bu, hem bir "metodolojik körlük" hem de "ölçek/rejim" boşluğu:** AGV literatürü sistem seviyesinde çalışıyor, simetri-RL literatürü ise tekil-robot/tekil-görev seviyesinde.

**Somut araştırma yönü önerisi:** Depo koridor/raf düzeninin (aisle/rack) doğal ayna-simetrisini açıkça formüle edip (Fikir 5'in metodolojisinde önerildiği gibi), bunu düşük seviye navigasyon politikasına E(2)-steerable katmanlar veya simetrik veri artırımı yoluyla kodlamak — literatürde şu ana kadar yapılmamış bir birleştirme.

---

## Gap 3: Aynalanmış/döndürülmüş harita düzeyinde genelleme nadiren doğrudan test ediliyor

**Kanıt ve ayrıntı:**
- **Kirk, Zhang, Grefenstette, Rocktäschel (2023, JAIR)** — "A Survey of Zero-shot Generalisation in Deep RL". Bu kapsamlı tarama, ortam varyasyonlarına (procedural generation, domain randomization) genellemeyi kategorize ediyor, ama **spesifik olarak mirror/rotasyon simetrisi altında genelleme** ayrı bir kategori olarak neredeyse hiç ele alınmıyor — daha çok görsel doku/renk/dinamik parametre varyasyonları üzerinden genelleme inceleniyor.
- **Theile ve ark. (2024)** kısmen yakın (eşdeğişken topluluklar simetrik haritalarda test ediliyor) ama sistematik "eğitimde görülmemiş aynalanmış harita" benchmark'ı kurmuyorlar; performans esas olarak örnek verimliliği (yakınsama hızı) üzerinden raporlanıyor, apayrı bir genelleme testi (unseen mirrored map) yapılmıyor.
- **Wang, Zhong, Chang, Allen-Blanchette (2025, LEGO)** — sürü robotik kontrolünde "takım büyüklüğü ve koordinat çerçevesi" değişikliklerine sıfır-ayarlı (zero-shot) transferi gösteriyorlar, ama bu görev-tipi simetrisi (permütasyon), harita/mekân simetrisi değil.

**Mekanizma analizi:** "Genelleme" araştırmaları genellikle örnekleme çeşitliliği (procedural generation, domain randomization) üzerinden ele alınıyor — bu örnekleme-temelli yaklaşım hesaplama açısından ucuz ve uygulaması kolay. Buna karşın "yapısal simetri garantili genelleme" (eşdeğişken mimari) araştırması matematiksel olarak daha zor ve mimari kısıtlamalar gerektiriyor; bu nedenle araştırmacılar genellikle ya tek yöntemi (randomization) ya da diğerini (equivariance) seçiyor, ikisini "mirror haritada zero-shot" gibi somut, dar bir test senaryosunda karşılaştırmıyorlar.

**Somut araştırma yönü önerisi:** Fikir 5'in önerdiği "Şekil 3: Aynalanmış/döndürülmüş haritalarda genelleme performansı ısı haritası" tam olarak bu boşluğu somut bir deneysel protokolle dolduruyor — literatürde bu şekilde raporlanmış bir çalışma bulunamadı.

---

## Gap 4: Sim-to-real açığının simetri-kırılma kaynaklarına ayrıştırılması hiçbir çalışmada yok

**Kanıt ve ayrıntı:**
- **Salvi, Ns, Lima, Kumar, Vatsal, Das (2025)** — humanoid lokomosyonda EMLP+PPO'nun vanilla PPO'ya kıyasla **daha düşük** yürüyüş kalitesi ve biyomekanik gerçekçilik gösterdiğini buluyorlar; "aşırı simetri kısıtları ifade gücünü sınırlayabilir" sonucuna varıyorlar. Bu, simetri varsayımının gerçek dünyada (veya gerçekçi biyomekanik hedeflerde) tam geçerli olmadığının dolaylı kanıtı — ama makale bu açığı nicel olarak simetri-kırılma kaynaklarına (hangi eklem, hangi sensör) ayrıştırmıyor.
- **Zhao, Queralta, Westerlund (2020, survey)** sim-to-real açığının genel kaynaklarını (dinamik model hatası, sensör gürültüsü, aktüatör gecikmesi) kategorize ediyor ama **simetri** bu kategorizasyonda hiç yer almıyor — çünkü survey 2020'de yazılmış, simetri-RL o dönemde henüz robotik sim-to-real ile kesişmemişti.
- **Peng ve ark. (2018)** dinamik randomizasyonun sim-to-real açığını kapattığını gösteriyor ama yöntem simetriden bağımsız; "hangi kaynaklar açığa katkıda bulunuyor" sorusuna simetri ekseninden hiç yaklaşmıyor.

**Mekanizma analizi:** Bu, alanın **kavramsal olarak henüz olgunlaşmamış** olmasından kaynaklanıyor — "simetri kırılması sim-to-real açığına ne kadar katkıda bulunuyor" sorusu, hem simetri-RL literatüründe (ki bu literatür ağırlıklı olarak simülasyon-içi çalışıyor) hem de sim-to-real literatüründe (ki bu literatür simetriyi bir faktör olarak düşünmüyor) kör nokta. İki literatürün kesişimi zayıf olduğu için, kesişimdeki bu spesifik soru hiç sorulmamış.

**Somut araştırma yönü önerisi:** Fikir 5'in metodoloji adım 5'i (fiziksel AGV'de başarı oranı + simetri-kırılma kaynaklarına göre hata ayrıştırması: asimetrik sensör gürültüsü vs. düzensiz zemin sürtünmesi) literatürde emsali bulunmayan, doğrudan özgün bir katkı olarak değerlendirilebilir.

---

## Gap 5: Veri artırımı vs. tam eşdeğişken ağ karşılaştırması, tekil-ajan 2D AGV navigasyonunda sistematik değil

**Kanıt ve ayrıntı:**
- **Zhou, Xiong, Zhao, Yan, Wei (2024, SymmQMIX)** — permütasyon-eşdeğişken ağ (EP2Net) + rotasyon/yansıma veri artırımını birleştirip QMIX'e entegre ediyorlar; %100 kata kadar örnek verimliliği artışı raporluyorlar. **Ancak** bu çok-ajanlı UAV yörünge/çizelgeleme problemi, tekil-ajan 2D işgal-ızgarası AGV navigasyonu değil.
- **Wang, Walters, Platt (2022, SO(2)-Equivariant RL)** tam eşdeğişken DQN/SAC'ı standart DQN/SAC'a karşı kıyaslıyor (manipülasyonda) — veri artırımı karşılaştırması yapmıyorlar (üçüncü kol yok).
- **McClellan ve ark. (2024/2025)** MARL bağlamında eşdeğişken ağların GNN'lere karşı üstünlüğünü gösteriyor ama yine tekil-ajan navigasyon değil, çok-ajan senaryosu.

**Mekanizma analizi:** Üç kollu (temel çizgi / veri artırımlı / tam eşdeğişken) sistematik karşılaştırma, hesaplama maliyeti yüksek bir deney tasarımı gerektiriyor (üç ayrı eğitim rejimi × birden fazla harita düzeni × birden fazla tohum). Çoğu çalışma kaynak kısıtları nedeniyle ikili karşılaştırmayla (temel çizgi vs. önerilen yöntem) sınırlı kalıyor. Fikir 5'in tasarımı (üç kollu karşılaştırma: Temel Çizgi PPO/SAC, Simetrik Veri Artırımlı PPO, Eşdeğişken Politika Ağı) literatürde nadir görülen bir titizlik seviyesi sunuyor.

**Bu gap'in Gap 1 ile ilişkisi:** Gap 5, Gap 1'in bir alt-bileşeni — sim-to-real'a taşınmadan önce bile, simülasyon-içi üç-kollu karşılaştırmanın kendisi eksik; Fikir 5 hem bu iç karşılaştırmayı hem de sim-to-real uzantısını aynı çalışmada birleştiriyor.

---

## Yöntem Ailesi Derinlemesine: Tam Eşdeğişken Ağ (escnn/E(2)-steerable, EMLP)

**Temsilci çalışmalar ve sayısal sonuçlar:**
- Wang, Walters, Platt (2022): Equivariant DQN/SAC, robotik manipülasyon görevlerinde temel çizgiye göre "önemli ölçüde daha örnek-verimli" (makalede spesifik kat sayısı görev bazlı değişiyor).
- Salvi ve ark. (2025): EMLP+PPO, humanoid 21-DoF lokomosyonda **örnek verimliliğini artırıyor** ama gait kalitesi/biyomekanik gerçekçilikte vanilla PPO'dan **geride kalıyor** — dezavantajın tersine döndüğü somut koşul: yüksek-serbestlik-dereceli, doğal hareketin tam simetrik olmadığı görevler (insan yürüyüşü de mükemmel simetrik değil, örn. baskın taraf etkileri).
- Theile ve ark. (2024): Özel eşdeğişken bileşen kullanmadan "eşdeğişken topluluk" tekniğiyle aynı faydayı elde etmeyi öneriyorlar — bu, escnn gibi kütüphanelerin "sınırlı bileşen kütüphanesi" dezavantajına karşı doğrudan bir yanıt.

**Diğer ailelerle kesişim:** Tam eşdeğişken ağlar, Gap 4'te tartışılan "aşırı kısıt" riskini taşıyor; bu risk kısmi/yaklaşık eşdeğişkenlik ailesinin (aşağıda) ortaya çıkış nedeni.

## Yöntem Ailesi Derinlemesine: Kısmi/Yaklaşık Eşdeğişkenlik

**Temsilci çalışmalar:**
- Chang ve ark. (2025, PI-MDP): Bellman geri-yayılımını simetrinin geçerli olduğu bölgelerde grup-değişmez, geçerli olmadığı bölgelerde standart uyguluyor — PE-DQN ve PE-SAC, Grid-World/lokomosyon/manipülasyonda temel çizgileri "önemli ölçüde" geçiyor.
- Park ve ark. (2024, Approximate Equivariance): Gevşetilmiş grup ve steerable konvolüsyonlarla yaklaşık eşdeğişken mimariler; tam simetri olduğunda tam eşdeğişken ağlarla eşit, yaklaşık simetri olduğunda onları geçiyor; ayrıca test-zamanı gürültüye karşı sağlamlık artışı gözlemleniyor (bu, sim-to-real bağlamında doğrudan ilgili bir yan bulgu).
- McClellan ve ark. (2025, PEnGUiN): Alt-grup eşdeğişkenliği, özellik-bazlı eşdeğişkenlik, bölgesel eşdeğişkenlik ve yaklaşık eşdeğişkenlik olarak dört tür kısmi eşdeğişkenlik tanımlıyorlar — tek bir çerçevede hem tam eşdeğişken hem tam eşdeğişken-olmayan temsilleri öğrenebiliyor.

**Bu ailenin Fikir 5 için önemi:** Fikir 5'in RQ3'ü (sim-to-real açığının simetri-kırılma kaynaklarına atfedilmesi) doğrudan bu ailenin teorik çerçevesine (özellikle PI-MDP) dayandırılabilir — zemin sürtünmesi/sensör asimetrisi, PI-MDP dilinde "yerel simetri kırılması" olarak formüle edilebilir.

---

## Yöntem Ailesi Derinlemesine: Sim-to-Real / Domain Randomization (simetrisiz temel çizgi ailesi)

**Temsilci çalışmalar ve rolleri:**
- Peng ve ark. (2018): Dinamik randomizasyon ile "reality gap" kapatma — 860+ atıfla alanın en temel referansı; Fikir 5'in temel çizgi sim-to-real yöntemi olarak kullanılabilir.
- Zhao, Queralta, Westerlund (2020): Kapsamlı sim-to-real survey (750+ atıf) — sim-to-real açığının standart kaynak kategorizasyonunu sağlıyor, Fikir 5'in "simetri-kırılma açığı" metriğinin bu standart kategorizasyona nasıl eklendiğini konumlandırmak için referans noktası.
- Loquercio ve ark. (2019): Drone racing'de domain randomization + CNN algı — yüksek hız/dinamik senaryoda sim-to-real'ın çalıştığını gösteren güçlü bir emsal, ama simetriden bağımsız.

**Diğer ailelerle kesişim ve çelişki:** Domain randomization ailesi ile eşdeğişkenlik ailesi **felsefi olarak zıt** stratejiler — randomization "her türlü varyasyona dayanıklı ol" derken, eşdeğişkenlik "belirli bir yapıyı (simetriyi) kesin olarak kullan" diyor. Literatürde bu iki stratejinin **birlikte** kullanıldığı (simetri + randomization hibrit) çalışma bulunamadı — bu da Fikir 5 için ek bir potansiyel katkı noktası olabilir (mevcut kapsamda zorunlu değil ama tartışma bölümünde değinilebilir).
