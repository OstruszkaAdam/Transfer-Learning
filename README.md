VI1 projekt – binární klasifikace metodou Transfer Learning
=====================================================

**Cílem projektu je automatické rozpoznání, zda rentgenový snímek patří do kategorie RA (snímky rukou s revmatoidní artritidou) nebo Bez RA (mohou občas vypadat podobně, ale artritidu nemají).** 

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

Ovládání – spouštění přes příkazovou řádku
---------
Aktuálně nefunguje a nejspíš nefungovalo už od začátku, je taky rozdíl mezi spouštěním na Windows a na Linuxu.

Windows:
```
cmd /C ""C:\Program Files\Python36\python.exe" C:\Users\Adam\PycharmProjects\Transfer_learning\test.py"

cmd /C ""C:\Program Files\Python36\python.exe" C:\Users\Adam\PycharmProjects\Transfer_learning\retrain.py"

```
Linux:
```
cd C:\Users\Adam\PycharmProjects\Transfer_learning

python retrain.py --bottleneck_dir="C:\Users\Adam\PycharmProjects\Transfer_learning" --how_many_training_steps=500 --model_dir="C:\Users\Adam\PycharmProjects\Transfer_learning" --output_graph="C:\Users\Adam\PycharmProjects\Transfer_learning\retrained_graph.pb" --output_labels="C:\Users\Adam\PycharmProjects\Transfer_learning\retrained_labels.txt" --image_dir="C:\Users\Adam\PycharmProjects\Transfer_learning\training_images"

C:\Users\Adam\PycharmProjects\Transfer_learning\classify_image_graph_def.pb

classify_image_graph_def.pb
```

Popis a výsledky provedených pokusů
---------
* K vytrénování posloužilo celkem 2505 snímků (624 bez příznaků RA a 1881 s příznaky), více jich počátkem ledna dostupných nebylo. Snímky zabírají několik GB a nejsou součástí tohoto repozitáře.
* Do jpg jsme alespoň převáděli s kvalitou 100 %, abychom artefakty v obrazech co nejvíc potlačili.
* Snímky byly kromě převodu formátu použity tak, jak byly pořízeny, nebylo potřeba vůbec žádné předzpracování (ořezání, rozřezání snímků obou rukou na prvou a levou, vyrovnání expozice, doostření, ani nic podobného). Program se musel vypořádat se snímky různě pootočenými, snímky pouze ruky nebo ruky i předloktí, snímky s prsteny, nemocničními náramky, amputovanými prsty i s mizernou technickou kvalitou (podexpozice).
* Vytrénování lze zvládnout i na průměrném stolním počítači za 5 až 30 minut v závislosti na výše zmiňovaných konstantách. Na průměrném notebooku to samé zabralo 20 až N minut. Při výchozím počtu 500 kroků zabere nejvíc času analýza snímků (vytvoření vektoru), teprve pak probíhá samotné učení se získanými hodotami, to už je ale celkem rychlé. Se všemi výše zmiňovanými konstantami na maximum trvá vytrénování asi 50 minut, přínosem ale je o 10 % vyšší spolehlivost klasifikace znímků bez RA.
* Na vzorkovém datasetu z počátku ledna celková přesnost učení kolísala od 96,3 % do 97,1 %. 
* Následné zařazení náhodného snímku do kategorie (soubor test.py) provedl program vždy správně se spolehlivostí od 80 % do 99,55 %. Jistější si byl u snímků spadajících do kategorie RA, kterých bylo v datasetu mnohem více. Snímky bez RA zařazoval s menší jistotou.
* Jedinou podmínkou je, že nové snímky k otestování musí mít stejně dobré rozlišení jako ty, na kterých se neuronka trénovala. Při testování na snímcích s nižním rozlišením a vyšší kompresí spadla spolehlivost na zhruba 70 %
* Zásluhy na dobrých výsledcích má právě metoda _Transfer Learning_. Díky ní lze snížit počet potřebných snímků ze statisíců až milionů na tisíce (náš případ) až desetitisíce. Metoda je založená na tom, že k vytrénování neuronky se použije volně dostupný _modul pro extrakci rysů obrázku_ z profesionálně vytrénovaných neuronek.

Hodilo by se
---------
- [ ] Rozjet spouštění přes příkazovou řádku – i s možností udělat několik spuštění za sebou a výsledky zaznamenávat do souborů (tak aby se nepřepisovaly)
- [x] Aby _test.py_ automaticky projel všechny snímky ve složce najednou bez zásahu uživatele
- [ ] Automatické mazaní složek před spuštěním učení (logy už se mažou, zbývají _bottleneck_data_)
- [ ] Zpracování alespoň souborů tiff nebo ideálně přímo dicom
- [ ] Vyzkoušet předzpracování snímků (ořez, převrácení, expozici nastavit na default ještě u dicomu) – mohla by se tím zvýšit přesnost klasifikace snímků bez RA


