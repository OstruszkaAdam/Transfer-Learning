VI1 projekt – klasifikace metodou Transfer Learning
=====================================================

**Cílem projektu je automatické rozpoznání, zda rentgenový snímek patří do kategorie RA (snímky rukou s revmatoidní artritidou) nebo Bez RA (mohou občas vypadat podobně, ale artritidu nemají). K tomuto účelu je kvůli omezenému množství dat použit _modul pro extrakci rysů obrázku_ založený na předtrénované neuronce Inception V3, ke které se pouze dotrénuje poslední vrstva (tzv. bottleneck).** 

K úspěšnému spuštění je potřeba
---------

-   Python STARŠÍ než 3.7 (jinak tensorflow nefunguje – na novější verze Pythonu
    ho ještě nepřizpůsobili).

-   Úspěšně nainstalovaný balíček tensorflow, cv2 (openCV), numpy a xlsxwriter.
    Při instalaci balíčku ne ručně, ale přes červenou žárovku v PyCharmu je
    řádově větší pravděpodobnost, že to skončí úspěchem.

-   Na Windows hromada štěstí, kupa času a pevné nervy (hlavně při pokusech o
    instalaci balíčků).

-   Na Linuxu se to obejde i bez štěstí, ale o to větší jsou nároky na čas a
    nervy (Zatímco na Windows je problém Python vůbec dostat, v Ubuntu jsou
    vestavěné rovnou dvě verze zároveň a proto červená žárovka zpočátku taky
    nefunguje)

Předzpracování snímků – nutný převod na jpg
---------

Program prozatím umí zpracovat pouze formát jpg, což se zrovna pro
medicínské použití nehodí, protože jpg je ztrátový formát. Při kvalitě 100% dochází ke ztrátě barevnosti. 
Při nižší kvalitě navíc vytváří v obrazu kompresní artefakty (vlastně obrazce, které tam nikdy nebyly).

Na Windows jsou dostupné programy úplně nefunkční nebo placené a tím pádem
funkční jen po dobu trvání trial verze.

Na Ubuntu Linuxu se jako nejvíc funkční možnost ukázal být vestavěný nástroj
Morgify. Pro převedení všech souborů ve složce jsem napsal jednoduchý prográmek
spouštěný přes příkazovou řádku.

**Krok 1:** Přesunout se v příkazové řádce do složky s Dicom soubory určenými k
převodu. Buď pomocí příkazu *cd* a našeptávání přes *Tab* nebo přes kontextovou
nabídku.

**Krok 2:** Překopírovat do této složky soubor DCMtoJPG (ze složky *Utils*) a
následně ho sputit příkazem

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sh DCMtoJPG.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ovládání programu 
---------

### Struktura složek

Řazení složek odpovídá tomu, jaká se posloupnost práce s daty. Google původně
toto nijak neřešil a byl v tom zmatek, ale teď by to mělo být alespoň trochu
intuitivní. Uživatel potřebuje věnovat pozornost pouze složkám 1 a 4, o zbytek
složek se stará systém.

#### 1_training_input_images

Sem zkopírujte složky se vstupními snímky, na kterých chcete neuronku trénovat.
Každou zde umístěnou složku bere neuronka jako samostatnou kategorii snímků.
Jako názvy kategorií slouží názvy složek. Nezapomeňte si pár snímků dát stranou
pro otestování funkčnosti neuronky. Poté už můžete spustit proces učení
(*retrain.py*)

#### 2_bottleneck_data

Tuto složku si naplní sama neuronka. Sem se umisťují už zpracované snímky, ze
kterých se neuronka učí. V případě použité neuronky *inceptionV3* to znamená, že
pro každý vstupní snímek se vytvoří jeho otisk – vektor s 2048 hodnotami.

#### 2_training_chache

Sem si program na počátku učení automaticky stáhne architekturu použitou
k trénování (tzn. zde je umístěna samotná neuronka *inceptionV3*). Během učení
se sem ukládají statistiky učícího procesu, které jsou zobrazitelné v nástroji
TensorBoard.

#### 2_training_output

Sem se ukládají výstupy z procesu učení.
Výsledný model, který po vytrénování slouží ke klasifikaci snímků, je v souboru _retrained_graph.pb_
Dále zde najdete parametry testování, uložené do Excelovské tabulky.
Při přetrénování neuronky se tyto soubory automaticky přepíšou.

#### 4_test_input_images

Sem umístěte snímky, na kterých chcete otestovat již vytrénovanou neuronku.

#### 5_test_chache

Během trénování se sem ukládají statistiky pro TensorBoard.

#### 6_test_output

Zde naleznete výstupy z testů. Každý test se uloží do samostatné složky, která
obsahuje jednotlivé snímky, do kterých program dopsal výslednou kategorii a
pravděpodobnost. Dále obsahuje souhrn výsledků testu v Excelovské tabulce.

### Spuštění programu
-   Trénování – otevřít a sputit soubor retrain.py

-   Testování – otevřít a sputit soubor test.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Před spuštěním učení (soubor retrain.py)

-   Naplnit složku *training_images* a několik málo snímků z ní přesunout do
    složky *test_images*. Tyto snímky slouží jen pro ruční testování otestování
    naučeného modelu. (Automatické testování si provádí program sám jako součást
    učení – v každém kroku učení rozdělí data na trénovací a testovací a provede
    otestování).

-   Vhodně nastavit konstanty HOW_MANY_TRAINING_STEPS a TRAIN_BATCH_SIZE a
    VALIDATION_BATCH_SIZE (jejich bližší popis je přímo v kódu). Pokud je
    dostatek výpočetního výkonu, dát třeba 4000 a 2000 a -1.

-   Do složky *bottleneck_data* si program pro každou trénovací fotku ukládá
    popisný vektor mající 2048 složek. Pokud jsme právě nepřesunuli nějaký
    snímek z trénovacích do testovacích, můžeme obsah složky ponechat. Pokud
    ano, je potřeba příslušný soubor s vektorem smazat (jinak zničíme princip
    testování, protože práci neuronky má smysl testovat jen na snímcích, které
    ještě neviděla). Při větších zásazích do snímků nutno smazat celou složku.

### Spouštění přes příkazovou řádku (nebo záložku Terminal v PyCharmu)

**Windows (funkční kromě předávání parametrů)**

Popis následujícího kódu: Nejprve je nutné přesunout se do složky s projektem a
potom ukázat cestu na Python a příslušný skript. Kdyby byl Python zadefinovaný v
systémových proměnných PATH, byly by příkazy trochu kratší. Aby to šlo spustit
na jednom řádku, bylo by potřeba předat v parametrech umístění všech složek, co
program používá.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cd C:\Users\Adam\PycharmProjects\Transfer_learning

cmd /C ""C:\Program Files\Python36\python.exe" retrain.py"
cmd /C ""C:\Program Files\Python36\python.exe" test.py"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux (funkční kromě předávání parametrů)**

V Linuxu je Python už zadefinovaný v systémových promměnných, takže místo cesty
k němu ho stačí prostě zavolat.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cd /home/ubuntu/PycharmProjects/Transfer-Learning/

python3 retrain.py
python3 test.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Předávání parametrů zatím nefunguje:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cd /home/ubuntu/PycharmProjects/Transfer-Learning/

python3 retrain.py --bottleneck_dir="/home/ubuntu/PycharmProjects/Transfer-Learning/1_training_input_images" --how_many_training_steps=10 --model_dir="C:\Users\Adam\PycharmProjects\Transfer_learning" --output_graph="C:\Users\Adam\PycharmProjects\Transfer_learning\retrained_graph.pb" --output_labels="C:\Users\Adam\PycharmProjects\Transfer_learning\retrained_labels.txt" --image_dir="C:\Users\Adam\PycharmProjects\Transfer_learning\training_images"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Spuštění TensorBoard pro sledování statistik procesu učení

TensorBoard slouží ke prohlížení vytvořených logů buď přímo živě během procesu
učení nebo u zpětně. V případě zájmu o sledování živě je potřeba TensorBoard
spustit následujícím příkazem (pouze na Linuxu) ještě před spuštěním učení.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tensorboard --logdir /home/ubuntu/PycharmProjects/Transfer-Learning/2_training_chache/tensorboard_logs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow na Windows má ještě více bugů než pro Linux a to platí i pro
TensorBoard. Na Windows se mi TensorBoard rozjet nepodařilo.

Hodilo by se
---------
**Předzpracování snímků**
- [ ] předělat skript na převod obrazových formátů tak, aby jako parametr přijímal cestu ke složce a šel spouštět odkudkoliv
- [ ] Pro zvýšení přesnosti klasifikace vyzkoušet předzpracování snímků (ořez, převrácení, expozici nastavit na default ještě u dicomu)
- [ ] Zpracování alespoň souborů tiff (s automatickým převodem z dicomu) nebo ideálně přímo dicom

**Testování snímků**
- [x] Aby _test.py_ automaticky projel všechny testovací snímky ve složce najednou bez zásahu uživatele
- [x] Ukládat výstupy jednotlivých testů do samostatných složek a výstupy formátovat tak, aby šly ve Wordu lehce převést na tabulku

**Technické a funkční záležitosti**
- [ ] Rozjet spouštění přes příkazovou řádku – i s možností udělat několik spuštění za sebou a výsledky zaznamenávat do oddělených složek a souborů (tak aby se nepřepisovaly) + souhrnně do jedné přehledné tabulky
- [X] Rozjet tensorboard (na Linuxu splněno, na Windows asi ztráta času)
- [ ] Porovnat současné fungování Tensorboard s původní sktrukturou složek a ověřit jeho správnou funkčnost
- [x] Automatické mazaní logů před spuštěním učení nebo testování
- [ ] Možnost automatického mazaní složky _2_bottleneck_data_

**Výstupy do Excelu**
- [ ] Nastavit jako desetinny oddelovat carku namisto tecky
- [ ] Vypis pravdepodobnosti pro jednotlive kategorie jednotne radit a naformatovat = dostat kategorie do zahlavi a do tabulky jen procenta
