# Literatür Taraması: Simetri-Farkında Derin Takviyeli Öğrenme ile Örnek-Verimli AGV Yol Planlaması ve Sim-to-Real Transfer

**Tarih:** 2026-09-14
**Kapsam:** Bu tarama, `Mission/Symmetry_Makale_Fikirleri.docx` içindeki **Makale Fikri 5**'i ("Symmetry-Exploiting Deep Reinforcement Learning for Sample-Efficient AGV Path Planning and Sim-to-Real Transfer") destekleyecek şekilde yürütülmüştür. Hedef dergi: **Symmetry (MDPI)**, Computer bölümü.

**Ayrıntılı gap & yöntem analizi için:** `simetri-farkinda-agv-rl-yontem-ve-gap-detay.md`

**Kaynak sayısı:** ~45 makale (arXiv API bu oturumda kalıcı olarak rate-limit'e takıldığı için tarama ağırlıklı olarak **Semantic Scholar** ve **OpenAlex** üzerinden çapraz doğrulanarak yapılmıştır; hedeflenen 55 yerine ~45 kaynağa ulaşılmıştır — kapsam genişliği bakımından yeterli ama tam hedefin biraz altında). Her bulgu en az 2 kaynakla desteklenmeye çalışılmıştır.

---

## 1. Yönetici Özeti

Simetri/eşdeğişkenlik-farkında (equivariant) takviyeli öğrenme, 2020'den itibaren hızla büyüyen ama hâlâ büyük ölçüde **konferans aşamasında** (ICLR, CoRL, NeurIPS, IROS) kalmış bir alt alan. AGV/AMR yol planlaması literatürü ise simetriyi neredeyse hiç açık bir tasarım parametresi olarak ele almıyor — çoğunlukla filo sevkiyatı/çizelgeleme RL'i veya klasik (A*, ACO, DWA) yöntemlerle sınırlı. Bu iki literatür arasındaki kesişim — **sim-to-real'a taşınan, harita simetrisini kullanan AGV navigasyon RL'i** — neredeyse boş. En yakın öncül çalışma (Theile ve ark., 2024, IROS) harita-tabanlı yol planlamasında eşdeğişken ağlar kullanıyor ama sim-to-real'a hiç geçmiyor. Bu, Makale Fikri 5'in önerdiği boşluğu doğrudan doğruluyor.

---

## 2. Trend Analizi

- **Büyüme eğrisi:** OpenAlex'te "symmetry-aware reinforcement learning robot navigation" geniş konu taraması 2016'da 15 eserden 2024'te 297'ye, 2025'te 431'e çıkıyor (bu sorgu geniş/gürültülü olduğundan mutlak sayılar değil eğim önemli — güçlü ve istikrarlı bir büyüme var).
- **Alt-alanın kendi zaman çizelgesi:** Taranan eşdeğişken-RL makalelerinin somut yayın yılları incelendiğinde, 2020 (MDP Homomorphic Networks, Mondal ve ark.) başlangıç noktası; 2022 gerçek bir sıçrama (Wang&Walters&Platt SO(2)-Equivariant RL, COVERS, On-Robot Equivariant Learning); **2024-2025 zirve** — taranan makalelerin yaklaşık %60'ı bu iki yılda yayınlanmış (Nguyen ve ark. 2024, Chang ve ark. 2025, Wang ve ark. 2025 LEGO, McClellan ve ark. 2025 PEnGUiN, Bousias ve ark. 2025, D'Elia ve ark. 2026, Salvi ve ark. 2025).
- **Venue kayması:** Alanın neredeyse tamamı ICLR/CoRL/NeurIPS/IROS gibi konferanslarda yayınlanıyor; **dergi düzeyinde, simetri temasını doğrudan iddia eden** bir makale örneği çok az bulundu. Bu, Symmetry (MDPI) için hem bir fırsat (az doldurulmuş niş) hem bir risk (hakemlerin RL-spesifik derinliğe aşina olmayabileceği) anlamına geliyor.
- **Yöntemsel kayma:** "Tam eşdeğişkenlik" (strict equivariance, escnn/EMLP tabanlı) yaklaşımından **"yaklaşık/kısmi eşdeğişkenlik"** (approximate/partial equivariance) yaklaşımına belirgin bir kayış var (Park ve ark. 2024; Chang ve ark. 2025 PI-MDP; McClellan ve ark. 2025 PEnGUiN) — alan, gerçek dünyada simetrinin nadiren tam olduğunu kabul ediyor. Bu kayış, Makale Fikri 5'in "sim-to-real simetri-kırılma açığı" metriğiyle doğrudan örtüşüyor.
- **Sim-to-real ayrı bir olgun literatür:** Domain randomization tabanlı sim-to-real transfer (Peng ve ark. 2018; Zhao ve ark. 2020 survey; Loquercio ve ark. 2019) çok daha olgun ve yüksek atıflı (700-860+ atıf), ama bu literatür simetri ile neredeyse hiç kesişmiyor — iki literatür birbirinden kopuk gelişmiş.

---

## 3. Gap (Boşluk) Analizi

*Ayrıntılı mekanizma analizi ve kanıt makaleleri için bkz. `simetri-farkinda-agv-rl-yontem-ve-gap-detay.md`.*

| # | Gap | Kanıt |
|---|-----|-------|
| G1 | Eşdeğişken/simetri-artırımlı DRL, **AGV/AMR sim-to-real transferi** ile hemen hemen hiç birleştirilmemiş | Theile ve ark. 2024 (yol planlaması, ama sadece simülasyon); Chang ve ark. 2025 (sim-to-real yok, Grid-World/lokomosyon/manipülasyon) |
| G2 | AGV/depo RL literatüründe simetri **açık bir tasarım parametresi olarak hiç kullanılmıyor** | Oyekanlu ve ark. 2020; Malus ve ark. 2020; Wesselhöft ve ark. 2022; Ellithy ve ark. 2024 — hiçbiri simetriden bahsetmiyor |
| G3 | Aynalanmış/döndürülmüş **harita** düzeyinde genelleme nadiren doğrudan test ediliyor (çoğu çalışma nesne/görev simetrisine odaklanıyor, ortam düzeni simetrisine değil) | Sadece Theile ve ark. 2024 kısmen yakın; Kirk ve ark. 2023 (zero-shot generalization survey) bunu bir açık problem olarak işaretliyor |
| G4 | "Sim-to-real açığının simetri-kırılma kaynaklarına ayrıştırılması" hiçbir çalışmada yok | Salvi ve ark. 2025'in "aşırı simetri kısıtı ifade gücünü sınırlıyor" bulgusu bu ayrıştırmanın gerekliliğine işaret ediyor ama kendisi yapmıyor |
| G5 | Veri artırımı vs. tam eşdeğişken ağ karşılaştırması MARL/manipülasyonda var ama **tekil-ajan 2D işgal-ızgarası AGV navigasyonunda sistematik olarak yok** | Zhou ve ark. 2024 (SymmQMIX, MARL); McClellan ve ark. 2024/2025 (MARL); Wang&Walters&Platt 2022 (manipülasyon) |

---

## 4. Yöntem / Materyal Karşılaştırması (Özet)

*Ayrıntılı satır-satır karşılaştırma için bkz. `simetri-farkinda-agv-rl-yontem-ve-gap-detay.md`.*

| Yöntem Ailesi | Temsilci Çalışmalar | İddia Edilen Avantaj | Gözlenen/İtiraf Edilen Dezavantaj |
|---|---|---|---|
| **Tam eşdeğişken ağ** (escnn/E(2)-steerable, EMLP) | Wang&Walters&Platt 2022; Salvi ve ark. 2025; Theile ve ark. 2024 | En güçlü örnek verimliliği + yapısal garanti | Sınırlı bileşen kütüphanesi (Theile ve ark. açıkça belirtiyor); aşırı kısıt ifade gücünü azaltabilir (Salvi ve ark. negatif bulgu) |
| **Simetrik veri artırımı** | Zhou ve ark. 2024 (SymmQMIX) | Basit, mevcut PPO/SAC'a kolayca eklenir | Eşdeğişkenliği garanti etmez; ağ mimarisiyle birleştirilmezse tek başına daha zayıf |
| **Kısmi/yaklaşık eşdeğişkenlik** | Park ve ark. 2024; Chang ve ark. 2025 (PI-MDP); McClellan ve ark. 2025 (PEnGUiN) | Gerçek dünyadaki kırılmış simetriye dayanıklı, iki dünyanın en iyisini birleştirir | Daha yeni, hiperparametre/mimari tasarımı daha karmaşık |
| **MDP homomorfizmi / temsil öğrenme** | van der Pol ve ark. 2020; Mavor-Parker ve ark. 2022 | Teorik olarak en genel çerçeve | Navigasyona doğrudan uygulama azınlıkta; daha soyut |
| **Domain randomization (simetrisiz temel çizgi)** | Peng ve ark. 2018; Loquercio ve ark. 2019; Zhao ve ark. 2020 (survey) | Olgun, kanıtlanmış sim-to-real yöntemi | Simetri önseli kullanmadığından örnek-verimsiz; Fikir 5'in karşılaştırma temel çizgisi olarak uygun |

---

## 5. Dergi Değerlendirmesi

Kullanıcının hedefi zaten **Symmetry (MDPI, ISSN 2073-8994)** olarak belirlenmiş durumda; bu bölüm sadece uygunluğu doğruluyor ve yedek seçenekleri not ediyor.

**Symmetry (MDPI) — OpenAlex venue verisi:**
- h-index: 125, 2 yıllık ortalama atıf: 2.42, 18.209 eser, Gold OA (Diamond değil), APC: 2.400 CHF (~2.896 USD)
- Dergideki baskın konular Fractional Differential Equations, Multi-Criteria Decision Making, Nonlinear Waves gibi matematik-ağırlıklı alanlar — RL/robotik makaleleri "Computer" bölümünde azınlıkta ama kabul ediliyor. **Sonuç: uygun ama hakemlerin RL-spesifik derinliğe aşina olmama riski var** — özet ve girişte simetri/asimetri bağlantısının ilk paragrafta açıkça kurulması (Mission dosyasındaki tavsiyeyle uyumlu) kritik.

**Yedek/alternatif dergiler (taramada AGV/AMR + RL makalelerinin yoğunlaştığı gözlenen, Gold/Diamond OA MDPI ve IEEE venue'ler):**
1. **Robotics** (MDPI, ISSN 2218-6581) — Wesselhöft ve ark. 2022 (AMR fleet RL survey), Aremu ve ark. 2026 (AMR path planning review) burada yayınlanmış; Gold OA.
2. **Sensors** (MDPI, ISSN 1424-8220) — çok sayıda AMR/DRL navigasyon makalesi (Zhang&Feng 2023 LM-SRNN, Sun ve ark. 2023 risk-aware DRL) burada; yüksek hacim, hızlı süreç.
3. **Machines** (MDPI, ISSN 2075-1702) — path planning ağırlıklı (Yang ve ark. 2023 review, Yang ve ark. 2022).

---

## 6. Kaynakça (taranan ~45 makale)

**Eşdeğişken/Simetri-Farkında RL — Teori & Manipülasyon**
1. van der Pol, Worrall, van Hoof, Oliehoek, Welling (2020). MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning.
2. Mondal, Nair, Siddiqi (2020). Group Equivariant Deep Reinforcement Learning.
3. Wang, Walters, Platt (2022). SO(2)-Equivariant Reinforcement Learning. ICLR.
4. Wang, Jia, Zhu, Walters, Platt (2022). On-Robot Learning With Equivariant Models. CoRL.
5. Liu, Xu, Huang, Liu, Oguchi, Zhao (2022). Continual Vision-based RL with Group Symmetries (COVERS). CoRL.
6. Tangri, Biza, Wang, Klee, Howell, Platt (2024). Equivariant Offline Reinforcement Learning.
7. Nguyen, Baisero, Klee, Wang, Platt, Amato (2024). Equivariant Reinforcement Learning under Partial Observability. CoRL.
8. Park, Bhatt, Zeng, Wong, Koppel, Ganesh, Walters (2024). Approximate Equivariance in Reinforcement Learning. AISTATS.
9. Park, Biza, Zhao, van de Meent, Walters (2022). Learning Symmetric Embeddings for Equivariant World Models.
10. Chang, Park, Seo, Horowitz, Lee, Choi (2025). Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments (PI-MDP).
11. Mavor-Parker, Sargent, Pehle, Banino, Griffin (2022). Using Forwards-Backwards Models to Approximate MDP Homomorphisms.
12. Ordoñez Apraez, Turrisi, Kostić, Martín, Agudo (2025). Morphological symmetries in robotics. IJRR.
13. Salvi, Ns, Lima, Kumar, Vatsal, Das (2025). Encoding Symmetries of Humanoid Robots using Equivariant Neural Networks in RL for Locomotion.
14. D'Elia, Zhan, Turrisi, Romualdi, Lerario, Camoriano, Pan, Pucci (2026). SKooP: Symmetric Koopman Predictions for Faster and More Generalizable Legged Robot Locomotion with RL.
15. Weissenbacher, Agarwal, Kawahara (2024). SiT: Symmetry-Invariant Transformers for Generalisation in Reinforcement Learning. ICML.
16. Marcos, Volpi, Komodakis, Tuia (2017). Rotation Equivariant Vector Field Networks. ICCV. *(görsel eşdeğişkenlik için temel referans)*

**Eşdeğişken/Simetrik Yol Planlaması — Doğrudan Öncül**
17. Theile, Cao, Caccamo, Sangiovanni-Vincentelli (2024). Equivariant Ensembles and Regularization for Reinforcement Learning in Map-based Path Planning. IROS. **(Fikir 5'in en yakın öncülü)**

**Çok-Ajanlı Simetrik/Eşdeğişken RL (sürü/filo ile ilgili)**
18. Hao, Hao, Mao, Wang, Yang, Li, Zheng, Wang (2023). Boosting Multiagent RL via Permutation Invariant and Permutation Equivariant Networks. ICLR.
19. McClellan, Haghani, Winder, Huang, Tokekar (2024). Boosting Sample Efficiency and Generalization in MARL via Equivariance (E2GN2). NeurIPS.
20. McClellan, Brothers, Huang, Tokekar (2025). PEnGUiN: Partially Equivariant Graph Neural Networks for Sample Efficient MARL.
21. Wang, Zhong, Chang, Allen-Blanchette (2025). Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control (LEGO).
22. Bousias, Pertigkiozoglou, Daniilidis, Pappas (2025). Symmetries-enhanced Multi-Agent Reinforcement Learning.
23. Tian, Yu, Qi, Wang, Feng, Wu, Shi, Luo (2024). Exploiting Hierarchical Symmetry in Multi-Agent Reinforcement Learning (HEPN). ECAI.
24. Zhou, Xiong, Zhao, Yan, Wei (2024). Symmetry-Augmented MARL for Scalable UAV Trajectory Design and User Scheduling (SymmQMIX). IEEE TMC.
25. Xu (2025). Symmetry-Driven CTDE: Enhancing Scalability and Sample Efficiency in MARL.

**Sim-to-Real Transfer & Domain Randomization**
26. Peng, Andrychowicz, Zaremba, Abbeel (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA.
27. Zhao, Queralta, Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey. IEEE SSCI.
28. Loquercio, Kaufmann, Ranftl, Dosovitskiy, Koltun (2019). Deep Drone Racing: From Simulation to Reality With Domain Randomization. IEEE T-RO.
29. Horváth, Erdős, Istenes, Horváth, Földi (2022). Object Detection Using Sim2Real Domain Randomization for Robotic Applications. IEEE T-RO.
30. Collins, Chand, Vanderkop, Howard (2021). A Review of Physics Simulators for Robotic Applications. IEEE Access.

**Genelleme / Zero-shot RL**
31. Kirk, Zhang, Grefenstette, Rocktäschel (2023). A Survey of Zero-shot Generalisation in Deep Reinforcement Learning. JAIR.
32. Zhu, Lin, Jain, Zhou (2023). Transfer Learning in Deep Reinforcement Learning: A Survey. IEEE TPAMI.

**AGV/AMR Yol Planlaması & Depo RL (uygulama alanı)**
33. Oyekanlu, Smith, Thomas, Mulroy, Hitesh, et al. (2020). A Review of Recent Advances in Automated Guided Vehicle Technologies. IEEE Access.
34. Malus, Kozjek, Vrabič (2020). Real-time order dispatching for a fleet of autonomous mobile robots using multi-agent RL. CIRP Annals.
35. Kozjek, Malus, Vrabič (2021). Reinforcement-Learning-Based Route Generation for Heavy-Traffic Autonomous Mobile Robot Systems. Sensors.
36. Bahrpeyma, Reichelt (2022). A review of the applications of multi-agent reinforcement learning in smart factories. Frontiers in Robotics and AI.
37. Ellithy, Salah, Fahim, Shalaby (2024). AGV and Industry 4.0 in warehouses: a comprehensive analysis and an innovative framework for flexible automation.
38. Wesselhöft, Hinckeldeyn, Kreutzfeldt (2022). Controlling Fleets of Autonomous Mobile Robots with Reinforcement Learning: A Brief Survey. Robotics (MDPI).
39. Tubis, Poturaj, Smok (2024). Interaction between a Human and an AGV System in a Shared Workspace — A Literature Review. Sustainability.
40. Leon, Li, Martin, Calvet, Panadero (2023). A Hybrid Simulation and Reinforcement Learning Algorithm for Enhancing Efficiency in Warehouse Operations. Algorithms.
41. Sun, Zhang, Yu, Zhang (2021). Motion Planning for Mobile Robots — Focusing on Deep Reinforcement Learning: A Systematic Review. IEEE Access.
42. Zhu, Wan Hasan, Ramli, Norsahperi, Kassim, Yao (2025). Deep Reinforcement Learning of Mobile Robot Navigation in Dynamic Environment: A Review. Sensors.
43. Aremu, Ahmed, El Ferik, Saif (2026). Autonomous Mobile Robot Path Planning Techniques — A Review: Metaheuristic and Cognitive Techniques. Robotics (MDPI).
44. Almazrouei, Kamel, Rabie (2023). Dynamic Obstacle Avoidance and Path Planning through Reinforcement Learning. Applied Sciences.

**Kalabalık/Sosyal Navigasyon DRL (metodolojik emsal, teğet)**
45. Chen, Liu, Kreiss, Alahi (2018). Crowd-Robot Interaction: Crowd-Aware Robot Navigation With Attention-Based Deep Reinforcement Learning. ICRA. *(en yüksek atıflı temel çalışmalardan biri, dikkat mekanizması emsali olarak)*

---

*Not: Bu taramada arXiv MCP servisi oturum boyunca kalıcı rate-limit (HTTP 429) hatası verdi; bu nedenle en güncel (son 1-2 ay) preprint'ler eksik kalmış olabilir. Semantic Scholar ve OpenAlex üzerinden erişilen makaleler arXiv preprint'lerini de (venue: "arXiv.org") kapsadığından kapsam ciddi ölçüde daraltılmamıştır, ancak bir sonraki iterasyonda arXiv'in doğrudan taranması önerilir.*
