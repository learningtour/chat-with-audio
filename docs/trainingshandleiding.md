# Trainingshandleiding — Chat with Audio

*Een praktijkcursus in zeven lessen, met oefenmateriaal. Voor iedereen die
audio wil verbeteren door erover te práten — podcastmakers, videomakers,
journalisten, docenten — zonder eerst een DAW te hoeven leren. (Deze
handleiding is bewust in het Nederlands; de rest van de documentatie is
Engels.)*

## Vooraf

**Wat je nodig hebt.** Een werkende installatie (zie
[Getting started](getting-started.md)), Claude Desktop of Claude Code, een
koptelefoon (echt — laptopspeakers verstoppen precies de problemen waar dit
over gaat), en het oefenmateriaal:

```bash
uv run python scripts/maak_oefenmateriaal.py ~/oefenmateriaal
```

Dat maakt zeven oefenbestanden, elk met één bekend, gedocumenteerd gebrek —
plus `ANTWOORDEN.md` (niet spieken vóór je zelf hebt geluisterd en
geanalyseerd).

**Het idee in één zin.** Jij beschrijft wat je hoort of wilt in gewone taal;
de assistent kiest het gereedschap, legt uit wat hij doet en waarom, en zet
elke bewerking als **sessie** klaar in de A/B-viewer zodat jij het laatste
woord hebt — met je oren.

**De gouden regel van de hele cursus:** vertrouw nooit een bewerking die je
niet A/B hebt geluisterd. De viewer speelt origineel en resultaat
sample-synchroon af; wisselen is één toetsaanslag. Alles in deze cursus
draait om die luistergewoonte.

---

## Les 1 — Kijken met je oren: analyseren

**Doel:** een bestand leren lezen vóór je er iets aan doet.

**Opdracht.** Zeg tegen Claude:

> Analyseer ~/oefenmateriaal/oefening-02-ruis.wav

**Waar kijk je naar in het antwoord?**

- **Scores (0-100)** voor loudness, ruis, dynamiek en helderheid — je eerste
  kompas. Alles boven ~80 is meestal prima; één lage score vertelt je waar
  het probleem zit.
- **Issues**: concrete bevindingen mét suggestie. De assistent stelt voor,
  jij beslist.
- **SNR (signaal-ruisverhouding)**: onder de ~20 dB hoor je de ruis duidelijk
  onder de spraak.

Open daarna de viewer ("open de viewer") en kijk naar het spectrogram:
constante ruis is een egale waas over het hele beeld.

**Controlevraag.** Wat is de SNR van dit bestand, en klopt dat met wat je
hoort in de pauzes tussen de zinnen?

---

## Les 2 — De één-knop-verbetering en het A/B-luisteren

**Doel:** de standaardworkflow leren: verbeteren → luisteren → bijsturen.

**Opdracht.**

> Maak oefening-02-ruis.wav beter

Open de sessie in de viewer en oefen dit ritme tot het een reflex is:

1. **Spatie** = afspelen. Je hoort B (het resultaat).
2. **Toets b** = wisselen tussen A en B, naadloos, op dezelfde plek.
3. **Toets r** = het **residu**: precies dát wat de bewerking heeft
   weggehaald. Dit is je belangrijkste kwaliteitscheck: hoor je in R alleen
   ruis, dan is de bewerking schoon; hoor je er spraak in, dan snoept de
   ontruising van de stem en moet het een tandje zachter.
4. **Toets x** = de **blinde luistertest**: X en Y zijn willekeurig
   origineel of bewerking, al het verklappende beeld vervaagt, en jij kiest
   eerlijk wat beter klinkt. Pas na je keuze zie je wat wat was. Doe dit bij
   twijfel áltijd — luider of "anders" klinkt snel beter dan het is; blind
   kiezen haalt die bias eruit. (A speelt sowieso loudness-matched af.)

**Bijsturen doe je in de chat**, in gewone woorden: "de s-klanken zijn nu te
dof" of "er zit nog ruis in de pauzes". De assistent past de keten aan en er
komt een nieuwe sessie naast te staan.

**Controlevraag.** Wat hoor je in het residu (R)? Alleen ruis, of ook een
beetje stem?

---

## Les 3 — Chirurgisch repareren: alleen dáár fixen waar het mis is

**Doel:** begrijpen wanneer je *niet* het hele bestand wilt behandelen.

Een koelkast die halverwege aanslaat verdient geen filter over de hele
opname — dat kost overal een beetje kwaliteit voor een probleem dat er maar
even is.

**Opdracht 1.**

> Fix oefening-01-brom.wav alleen waar iets mis is

Kijk in de viewer naar de tijdlijnbalk **ingrepen**: daar zie je wáár is
ingegrepen (klik erop om ernaartoe te springen). Alles buiten de regio's is
bit-voor-bit onaangetast. In het chatantwoord staat ook de **tweede meetpas**
(`verification`): de detectoren zijn na afloop opnieuw over het resultaat
gelopen en melden eerlijk of het probleem echt weg is.

**Opdracht 2.** Zelfde met `oefening-03-clipping.wav` (afgekapte toppen) en
`oefening-04-dreun.wav` (passerende vrachtwagen). Let bij de dreun op: wordt
de stém dunner van de fix? (Wissel A/B precies tijdens de dreun.)

**Goed om te weten.** Eén fysiek probleem kan meerdere detecties geven — een
brom of dreun tilt in de spreekpauzes ook de gemeten ruisvloer op, dus naast
"netbrom" kan er ook een ruisregio gemeld worden. Dat is meten, geen dubbel
behandelen.

**Controlevraag.** Hoeveel seconden van oefening-01 zijn behandeld, en
hoeveel bleven onaangetast?

---

## Les 4 — Tekstmontage: knippen op woorden

**Doel:** monteren zonder golfvormen: op het transcript.

**Opdracht 1.**

> Maak de pauzes in oefening-05-pauzes.wav strakker, maximaal 0,6 seconde —
> laat me eerst het plan zien

Dat "laat me eerst het plan zien" is `apply=False`: je krijgt de montagelijst
met per knip de transcriptcontext, en de viewer toont de voorgenomen knips op
de tijdlijn — vóór er iets gebeurt. Bevalt het plan, dan: "voer het uit".

**Opdracht 2 (eigen materiaal).** Neem een spraakmemo op van een halve
minuut waarin je bewust wat "uhm's" en een verspreking stopt. Dan:

> Haal de uhs en verdubbelingen eruit en maak de pauzes strakker

En voor redactie/privacy:

> Bliep elke keer dat ik [naam] zeg

**Let op.** Knips maken het bestand korter, dus de A/B in de viewer loopt na
de eerste knip uit de pas — de tijdlijn toont de knips op de oorspronkelijke
tijdlijn. Exporteer de kniplijst desgewenst als DAW-markers
("exporteer de markers").

**Controlevraag.** Hoeveel seconden won de pauze-aanscherping, en klinken de
lassen onhoorbaar? (Luister op de knippunten!)

---

## Les 5 — Maten en normen: loudness en compliance

**Doel:** snappen waaróm audio een loudness-norm heeft, en ernaartoe werken.

**Achtergrond in drie zinnen.** Streamingdiensten en omroepen normaliseren
op **LUFS** (waargenomen luidheid over het geheel); mik je te laag dan wordt
je programma zachter afgespeeld dan de rest, te hoog dan wordt het
teruggedraaid én klinkt het platgeslagen. **True peak** (dBTP) bewaakt de
pieken tússen de samples, die na lossy-compressie kunnen gaan clippen.
Daarom eisen specs bijvoorbeeld -16 LUFS / -1 dBTP (podcast) of -23 LUFS /
-1 dBTP (EBU-omroep).

**Opdracht.**

> Breng oefening-06-te-stil.wav naar podcastniveau en check het tegen de
> Apple Podcasts-spec

Bekijk het compliance-paneel in de viewer: per criterium gemeten vs vereist.
Daarna:

> Wat doet mp3-compressie straks met deze master?

(`codec_preview`) — let op de true peak ná de codec: dát is waarom het
plafond onder de 0 moet blijven.

**Controlevraag.** Hoeveel dB is het bestand opgetild, en waarom staat de
limiter-ceiling nét onder het true-peak-doel?

---

## Les 6 — De eindtoets

**Doel:** alles combineren, in de goede volgorde.

**Opdracht.**

> Ik wil eindtoets.wav publicatieklaar hebben op -16 LUFS. Analyseer eerst,
> vertel me je plan, en voer het pas uit als ik akkoord ben.

Er zit brom in, opkomende ruis (subtiel — luister in de pauzes), zware
clipping, en de spraak is veel te stil. Twee leerpunten zitten verstopt:

1. **Volgorde doet ertoe**: eerst repareren (chirurgisch), dán normaliseren.
   Wie eerst normaliseert, versterkt de gebreken mee.
2. **Metingen kunnen liegen door defecten**: de clip-knal trekt de
   integrated-loudness-meting omhoog; na de reparatie klopt de meting weer.

Sluit af met een QC-rapport ("maak een QC-rapport met de apple-podcast-spec")
en — belangrijkste van alles — een blinde luistertest tegen het origineel.

**Geslaagd als:** het rapport groen is én je in de blinde test zelf de
bewerking kiest.

---

## Les 7 — Voor gevorderden: eigen materiaal en de rest van de kist

Vanaf hier is het echte werk. Een greep, met de letterlijke vraag erbij:

| Situatie | Vraag aan de chat |
|---|---|
| Interview met twee recorders | "Sync deze twee opnames en mix mijn boom en lav automatisch" |
| ADR/voice-over moet in een scène passen | "Laat deze studio-opname klinken alsof hij in dezelfde ruimte is opgenomen als scene.wav" |
| Stem 'aan de telefoon' | "Pas het telefoon-recept toe op dit spoor" |
| Aflevering wegsturen | "Maak een afleverpakket met QC-rapport en checksums" |
| Podcast met hoofdstukken | "Exporteer als mp3 met hoofdstukken op de onderwerpwissels" |
| Zelfde klank als vorige aflevering | "Laat dit klinken als aflevering-12-master.wav" |
| Iets werkt goed | "Bewaar deze keten als recept 'mijn-podcast'" |
| Terugkerende twijfel | "Doe een blinde test" — en kies met je oren |

**Werkgewoontes van de pro's:**

- **Praat in problemen, niet in oplossingen.** "De stem klinkt dun als de
  vrachtwagen voorbijkomt" geeft de assistent meer om mee te werken dan
  "zet een EQ op 200 Hz".
- **Residu luisteren (r)** na elke ontruising of de-esser.
- **Blind kiezen (x)** bij elke twijfel.
- **Sessies opruimen**: "laat mijn sessies zien" / "ruim sessies ouder dan
  30 dagen op" (er komt altijd eerst een oefenrun die toont wat er weg zou
  gaan).
- **Niet blijven stapelen**: twee, drie gerichte ingrepen zijn bijna altijd
  beter dan zeven halve. Begin liever opnieuw vanaf het origineel met een
  beter verzoek.

---

## Spiekbrief: woordenlijst

| Term | Betekenis |
|---|---|
| **LUFS** | Luidheid zoals het gehoor die ervaart, gemeten over het geheel (BS.1770). Podcast ≈ -16, EBU-tv -23, muziekstreaming ≈ -14 |
| **True peak (dBTP)** | Piekniveau inclusief de pieken tússen samples; bewaakt clipping na lossy-codecs |
| **SNR** | Afstand tussen spraak en ruisvloer in dB; onder ~20 dB is ruis storend hoorbaar |
| **Residu (R)** | Het verschil origineel − bewerking: precies wat er is weggehaald |
| **Notch** | Zeer smal filter dat één frequentie (brom) wegprikt zonder de rest te raken |
| **Crossfade** | Overlappende in-/uitfade op een las zodat je de knip niet hoort |
| **RT60** | Nagalmtijd: hoelang een ruimte doet over 60 dB uitsterven |
| **Stems** | De losse lagen van een mix (zang/drums/bas/rest, of dialoog/muziek/effecten) |
| **Dither** | Bewust minuscule ruis bij 16-bit-export; voorkomt kwantisatievervorming op zachte staarten |
| **Compliance** | Pass/fail-check tegen een afleverspec (EBU R128, Netflix, ACX, …) |

*Antwoorden op de controlevragen staan — met de bestandsgebreken — in
`ANTWOORDEN.md` bij het oefenmateriaal. De volledige toolreferentie:
[tools.md](tools.md); alle workflows: [workflows.md](workflows.md).*
