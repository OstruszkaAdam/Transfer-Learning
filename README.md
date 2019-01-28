VI1 projekt – klasifikace metodou Transfer Learning
=====================================================

**Cílem projektu je automatické rozpoznání, zda rentgenový snímek patří do kategorie RA (snímky rukou s revmatoidní artritidou) nebo Bez RA (mohou občas vypadat podobně, ale artritidu nemají). K tomuto účelu je kvůli omezenému množství dat použit _modul pro extrakci rysů obrázku_ založený na předtrénované neuronce Inception V3, ke které se pouze dotrénuje poslední vrstva (tzv. bottleneck).** 

K úspěšnému spuštění je potřeba
---------
* Python STARŠÍ než 3.7 (jinak tensorflow nefunguje)
* Úspěšně nainstalovaný balíček tensorflow, cv2 (openCV) a numpy. Při instalaci balíčku přes červenou žárovku v PyCharmu je řádově větší pravděpodobnost, že se to povede
* Na Windows hromada štěstí, kupa času a pevné nervy (hlavně při pokusech o instalaci balíčků)
* Na Linuxu se to obejde i bez štěstí, ale o to větší jsou nároky na čas a nervy (Zatímco na Windows je problém Python vůbec dostat, v Ubuntu jsou vestavěné rovnou dvě verze zároveň a proto červená žárovka zpočátku taky nefunguje)

Ovládání – před spuštěním učení (soubor retrain.py)
---------
* Naplnit složku _training_images_ a několik málo snímků z ní přesunout do složky _test_images_. Tyto snímky slouží jen pro ruční testování otestování naučeného modelu. (Automatické testování si provádí program sám jako součást učení – v každém kroku učení rozdělí data na trénovací a testovací a provede otestování).
* Vhodně nastavit konstanty HOW_MANY_TRAINING_STEPS a TRAIN_BATCH_SIZE a VALIDATION_BATCH_SIZE (jejich bližší popis je přímo v kódu). Pokud je dostatek výpočetního výkonu, dát třeba 4000 a 2000 a -1.
* ~~Vždy smazat složku _tensorboard_logs_, protože program si po sobě zatím neuklízí a s plnou nejde učení spustit.~~ Program už ji umí smazat sám.
* Do složky _bottleneck_data_ si program pro každou trénovací fotku ukládá popisný vektor mající 2048 složek. Pokud jsme právě nepřesunuli nějaký snímek z trénovacích do testovacích, můžeme obsah složky ponechat. Pokud ano, je potřeba příslušný soubor s vektorem smazat (jinak zničíme princip testování, protože práci neuronky má smysl testovat jen na snímcích, které ještě neviděla). Při větších zásazích do snímků nutno smazat celou složku.
* Program prozatím umí zpracovat pouze formát jpg, což se zrovna pro medicínské použití nehodí, protože jpg je ztrátový formát a vytváří v obraze kompresní artefakty (vlastně obrazce, které tam nikdy nebyly).
* Výsledný model, který po vytrénování slouží ke klasifikaci snímků, je uložený v souboru _retrained_graph_ a mazat se nejspíš nemusí.

Ovládání – spouštění přes příkazovou řádku (nebo záložku Terminal v PyCharmu)
---------
**Windows (funkční kromě předávání parametrů)**

Popis následujícího kódu: Nejprve je nutné přesunout se do složky s projektem a potom ukázat cestu na Python a příslušný skript. Kdyby byl Python zadefinovaný v systémových proměnných PATH, byly by příkazy trochu kratší. Aby to šlo spustit na jednom řádku, bylo by potřeba předat v parametrech umístění všech složek, co program používá.
```
cd C:\Users\Adam\PycharmProjects\Transfer_learning

cmd /C ""C:\Program Files\Python36\python.exe" retrain.py"
cmd /C ""C:\Program Files\Python36\python.exe" test.py"

```
 
**Linux (funkční kromě předávání parametrů)**

V Linuxu je Python už zadefinovaný v systémových promměnných, takže místo cesty k němu ho stačí prostě zavolat.
```
cd /home/ubuntu/PycharmProjects/Transfer-Learning/

python3 retrain.py
python3 test.py
```

Předávání parametrů zatím nefunguje:
```
cd /home/ubuntu/PycharmProjects/Transfer-Learning/

python3 retrain.py --bottleneck_dir="/home/ubuntu/PycharmProjects/Transfer-Learning/1_training_input_images" --how_many_training_steps=10 --model_dir="C:\Users\Adam\PycharmProjects\Transfer_learning" --output_graph="C:\Users\Adam\PycharmProjects\Transfer_learning\retrained_graph.pb" --output_labels="C:\Users\Adam\PycharmProjects\Transfer_learning\retrained_labels.txt" --image_dir="C:\Users\Adam\PycharmProjects\Transfer_learning\training_images"

```

Spuštění TensorBoard pro sledování statistik procesu učení
---------
TensorBoard slouží ke prohlížení vytvořených logů buď přímo živě během procesu učení nebo u zpětně.
V případě zájmu o sledování živě je potřeba TensorBoard spustit následujícím příkazem (pouze na Linuxu) ještě před spuštěním učení.
```
tensorboard --logdir /home/ubuntu/PycharmProjects/Transfer-Learning/2_training_chache/tensorboard_logs
```
TensorFlow na Windows má ještě více bugů než pro Linux a to platí i pro TensorBoard. Na Windows se mi TensorBoard rozjet nepodařilo.

Popis a výsledky provedených pokusů
---------
* K vytrénování posloužilo celkem 2505 snímků (624 bez příznaků RA a 1881 s příznaky), více jich počátkem ledna dostupných nebylo. Snímky zabírají několik GB a nejsou součástí tohoto repozitáře.
* Do jpg jsme alespoň převáděli s kvalitou 100 %, abychom artefakty v obrazech co nejvíc potlačili.
* Snímky byly kromě převodu formátu použity tak, jak byly pořízeny, nebylo potřeba vůbec žádné předzpracování (ořezání, rozřezání snímků obou rukou na pravou a levou, vyrovnání expozice, doostření, ani nic podobného). Program se musel vypořádat se snímky různě pootočenými, snímky pouze ruky nebo ruky i předloktí, snímky s prsteny, nemocničními náramky, amputovanými prsty i s mizernou technickou kvalitou (podexpozice).
* Vytrénování lze zvládnout i na průměrném stolním počítači za 5 až 30 minut v závislosti na výše zmiňovaných konstantách. Na průměrném notebooku to samé zabralo 20 až N minut. Při výchozím počtu 500 kroků zabere nejvíc času analýza snímků (vytvoření vektoru), teprve pak probíhá samotné učení se získanými hodotami, to už je ale celkem rychlé. Se všemi výše zmiňovanými konstantami na maximum trvá vytrénování asi 50 minut, přínosem ale je o 10 % vyšší spolehlivost klasifikace znímků bez RA.
* Na vzorkovém datasetu z počátku ledna celková přesnost učení kolísala od 96,3 % do 97,1 %. 
* Následné zařazení náhodného snímku do kategorie (soubor test.py) provedl program vždy správně se spolehlivostí od 80 % do 99,55 %. Jistější si byl u snímků spadajících do kategorie RA, kterých bylo v datasetu mnohem více. Snímky bez RA zařazoval s menší jistotou.
* Jedinou podmínkou je, že nové snímky k otestování musí mít stejně dobré rozlišení jako ty, na kterých se neuronka trénovala. Při testování na snímcích s nižním rozlišením a vyšší kompresí spadla spolehlivost na zhruba 70 %
* Zásluhy na dobrých výsledcích má právě metoda _Transfer Learning_. Díky ní lze snížit počet potřebných snímků ze statisíců až milionů na tisíce (náš případ) až desetitisíce. Metoda je založená na tom, že k vytrénování neuronky se použije volně dostupný _modul pro extrakci rysů obrázku_ z profesionálně vytrénovaných neuronek.

Hodilo by se
---------
- [ ] Ukládat výstupy jednotlivých testů do samostatných složek a výstupy formátovat tak, aby šly ve Wordu lehce převést na tabulku
- [ ] Rozjet spouštění přes příkazovou řádku – i s možností udělat několik spuštění za sebou a výsledky zaznamenávat do souborů (tak aby se nepřepisovaly)
- [ ] Rozjet tensorboard, porovnat jeho současné fungování s původní sktrukturou složek
- [x] Aby _test.py_ automaticky projel všechny testovací snímky ve složce najednou bez zásahu uživatele
- [x] Automatické mazaní logů před spuštěním učení nebo testování
- [ ] Možnost automatického mazaní složky _2_bottleneck_data_
- [ ] Zpracování alespoň souborů tiff (s automatickým převodem z dicomu) nebo ideálně přímo dicom
- [ ] Pro zvýšení přesnosti klasifikace vyzkoušet předzpracování snímků (ořez, převrácení, expozici nastavit na default ještě u dicomu)


