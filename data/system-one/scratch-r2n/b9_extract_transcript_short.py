# General: extraction (25), transcription-adjacent (25), short one-liners / greetings / follow-ups (30).
ROWS = [
# --- extraction ---
dict(cat="extraction", lang="en", sys=None, prompt="""Pull out every date and what happens on it:

The survey is booked for the 4th, the report should land a week later, and the planning deadline is the 19th. If the report slips past the 12th we lose the slot and the next committee is on the 3rd of next month. The structural engineer is away from the 8th to the 15th and has asked for anything he needs to sign before he goes. Building control were told we would submit in the first week of the month after that, which now looks optimistic, and the lease on the temporary office runs out at the end of the quarter."""),
dict(cat="extraction", lang="en", sys=None, prompt="""List the people mentioned here and what each is responsible for:

Hulda is handling the lease, Petur has the insurance quotes, and the fit-out is with an external firm that Marta found. Bjorn signs anything over two million."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Extract the quantities and units as a table:

For the render we need roughly 18 bags of sand, six bags of cement, 40 litres of primer and about 12 square metres of mesh, plus a spare bag of each."""),
dict(cat="extraction", lang="en", sys="You extract structured facts and mark anything uncertain with a question mark.", prompt="""Get me the company, role and dates from this CV line: 'Nordvik Logistics — warehouse systems lead, Rotterdam, 2019 to early 2023 (maternity cover from 2021)'."""),
dict(cat="extraction", lang="en", sys=None, prompt="""From this email, list only the action items assigned to me:

Marta will send the figures. Could you check the supplier contract for the notice period and confirm whether we can move the November delivery? Bjorn is booking the venue. I will chase the printer."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Extract the prices and what they cover:

The base package is 84,000 a year including support during office hours. Out of hours support is another 26,000. Onboarding is a one-off 150,000 but they waive half of it if you sign for two years."""),
dict(cat="extraction", lang="en", sys=None, prompt="""List every measurement in this description and say which ones are internal versus external:

The cabinet is 1,800 tall and 600 deep, with three shelves at 320 clear height. The plinth adds 80 and the back panel takes 12 off the depth."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Find the contact details in this signature block and label them:

Gudrun Sveinsdottir | Head of Operations | Vesturhofn ehf | +354 555 2211 | gudrun@vesturhofn.is | Tryggvagata 11, 101 Reykjavik"""),
dict(cat="extraction", lang="en", sys=None, prompt="""Extract the conditions under which the discount applies:

New customers get fifteen percent off the first year if they sign before the end of the quarter and commit to at least ten seats. The discount does not apply to the onboarding fee or to reseller contracts."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Which requirements here are hard and which are preferences?

The role needs five years of backend experience and the right to work here. Ideally the person has led a small team, and we would like someone comfortable in Go, though we will teach it."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Pull the times and platforms out of this and put them in order:

The 07:12 goes from platform 3, then there is nothing until 08:40 from platform 1, and the 09:05 leaves from 3 again but only on weekdays."""),
dict(cat="extraction", lang="en", sys=None, prompt="""List every risk this paragraph mentions, one line each:

If the crane booking moves we lose two days, and the concrete pour has to happen before the frost. The subcontractor has one other job that could overrun, and we have no second supplier for the cladding. The site access agreement with the neighbour expires in November and has not been renewed. Two of the four scaffolders are agency staff whose contracts end this month, and the weather forecast for the pour week is already marginal, with overnight temperatures near zero."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Extract the numbers and what they measure:

Retention went from 41 to 47 percent, sessions per week fell slightly to 3.2, and support contacts dropped by about a fifth, though we changed the help page in the same period."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Identify the decision, who made it and when, from this note: 'Agreed on Tuesday's call — Anna confirmed we go with the fixed price option, effective from the next invoice.'"""),
dict(cat="extraction", lang="en", sys=None, prompt="""From this listing, extract the facts a buyer would want: 'Ground floor, 68 sqm, two bedrooms, built 1974, communal laundry, shared garden, no parking space, service charge 24,000 a month.'"""),
dict(cat="extraction", lang="en", sys=None, prompt="""Which of these are questions that need an answer, and which are statements? List them separately:

We are closing the ticket. Did the customer confirm the address? The refund went out on Friday. Should we tell support before or after?"""),
dict(cat="extraction", lang="en", sys=None, prompt="""Extract the ingredients with quantities from this paragraph and leave the method out:

Soften two onions in a good knob of butter, add 400 grams of diced lamb and brown it, then a tablespoon of flour, 500 ml of stock and a bay leaf."""),
dict(cat="extraction", lang="en", sys=None, prompt="""Find the version numbers and what they apply to: 'We are on 14.2 in production, staging runs 15.0-rc2, and the client library is pinned at 3.9.4 because 4.x drops Node 18.'"""),
dict(cat="extraction", lang="de", sys=None, prompt="""Liste alle Termine und Fristen aus diesem Absatz auf:

Die Unterlagen muessen bis zum 12. eingereicht werden, die Pruefung findet in der Woche darauf statt, und das Ergebnis kommt spaetestens Ende des Monats."""),
dict(cat="extraction", lang="fr", sys=None, prompt="""Relevez les montants et ce qu'ils couvrent: 'L'abonnement est de 49 euros par mois, plus 120 euros d'installation, et 15 euros par utilisateur supplementaire au-dela de dix.'"""),
dict(cat="extraction", lang="es", sys=None, prompt="""Extrae los datos del envio: 'Pedido 7741, salio de Valencia el 3, entrega prevista el 7, transportista Seur, peso 12,4 kg, dos bultos.'"""),
dict(cat="extraction", lang="ja", sys=None, prompt="""次の文から、日付と担当者と作業内容を取り出してください。「十月七日に佐藤さんが在庫確認、九日に山田さんが発注、月末に棚卸し。」"""),
dict(cat="extraction", lang="pt", sys=None, prompt="""Liste as obrigacoes do inquilino que aparecem neste texto: 'O inquilino deve pagar a renda ate ao dia oito, manter o imovel em bom estado e avisar com sessenta dias de antecedencia.'"""),
dict(cat="extraction", lang="nl", sys=None, prompt="""Haal de afspraken en de namen uit deze notitie: 'Jeroen belt de leverancier, Sanne maakt de offerte af voor vrijdag, en we beslissen maandag in het overleg.'"""),
dict(cat="extraction", lang="is", sys=None, prompt="""Taktu ut dagsetningar og verkefni ur thessum texta: 'Uttektin er 14. mars, skyrslan a ad berast viku sidar og fundur nefndarinnar er 2. april.'"""),

# --- transcription adjacent ---
dict(cat="transcript", lang="en", sys=None, prompt="""Punctuate this dictated note without changing any words:

right so the delivery is thursday morning between eight and twelve someone has to be in i can do it if you take the kids to swimming otherwise we move it to saturday but saturday costs more and they said the driver rings twenty minutes before so there is no point sitting by the door all morning also they cannot bring it up the stairs so we need someone else for that i asked and they were quite clear about it"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Clean up this voice note into a short paragraph, keep my voice:

um so i spoke to the the landlord and he says the the boiler is covered but the radiators aren't which i think is what we expected anyway so um probably we just get someone in ourselves"""),
dict(cat="transcript", lang="en", sys="You tidy dictation. Never add facts, never change names or numbers.", prompt="""Turn this into a clean note:

call kristin about the invoice the one from september not the new one and also ask about the credit note she mentioned i think it was four hundred and something"""),
dict(cat="transcript", lang="en", sys=None, prompt="""This transcript has obvious mishearings. Fix the ones you are confident about and flag the rest:

the committee approved the new bee law on cycle lanes and asked the planing officer to report back in the autumn"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Turn this dictated shopping list into a tidy list grouped sensibly:

milk bread eggs oh and the the coffee we like the dark one apples bananas washing up liquid and those oat things for the kids"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Split this run-on transcript into speakers, it was two people:

so how did the site visit go fine mostly the roof is worse than the photos did they give a number not yet next week they said"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Remove the filler and false starts but keep every fact:

yeah so we we tried the new supplier in in spring and it was fine i mean it was fine but the lead time was like three weeks longer than they said and then when we when we asked about it they blamed the ferry which might even be true but nobody told us until we chased them twice so the the quality was good the the price was good but i would not put them on anything that has a date attached to it"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Make bullet notes from this recorded ramble:

remember to call the plumber about the radiator in the back room the car needs an oil change before the trip and i said i'd send petur the photos from saturday"""),
dict(cat="transcript", lang="en", sys=None, prompt="""My phone transcribed a meeting and mangled the names. Here is a line — how would you guess at the real names? 'Then Good Run said she would speak to Bury Oleifson about the lease.'"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Turn this dictation into a proper email to a colleague, two short paragraphs:

hi anna quick one the figures for q3 came in lower than we thought mostly the shipping cost can we talk tomorrow before the board pack goes out thanks"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Add paragraph breaks and punctuation to this, nothing else:

we arrived about seven the ferry was late again and the wind had got up by the time we walked to the guesthouse everything was shut so we ate crisps for dinner and slept badly"""),
dict(cat="transcript", lang="en", sys=None, prompt="""This is an automatic caption file. Rewrite it as readable prose:

00:01 so today we're looking at
00:04 the way the current runs past the island
00:08 and why the fishing is better on the north side"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Condense this rambling voicemail into two sentences I can forward:

hi it's me again sorry about earlier so the thing with the booking is that they only hold it until five today and i can't call them from the site so if you could that would be great the reference is on the fridge i think or maybe in the email from tuesday the one with the attachment and if they ask we want the later slot not the one at half six because of the trains oh and they need a card number to hold it"""),
dict(cat="transcript", lang="en", sys=None, prompt="""Number the action items you can hear in this dictated note and ignore the asides:

ok so first thing print the labels second call the courier about the pickup window they said before eleven oh and remind me to buy tape we are nearly out"""),
dict(cat="transcript", lang="en", sys=None, prompt="""How would you punctuate a dictated sentence where the speaker clearly changed direction halfway? Here is one: 'and then we went to the no wait it was the tuesday we went to the harbour first'"""),
dict(cat="transcript", lang="de", sys=None, prompt="""Setz bitte Satzzeichen in dieses Diktat, ohne Woerter zu aendern:

also der termin am dienstag passt nicht ich bin da noch unterwegs mittwoch vormittag ginge oder donnerstag nach zwei sag bescheid"""),
dict(cat="transcript", lang="fr", sys=None, prompt="""Corrigez la ponctuation de cette transcription sans modifier les mots:

le train part a sept heures dix il faut etre la vingt minutes avant sinon ils ferment les portes j ai deja les billets"""),
dict(cat="transcript", lang="es", sys=None, prompt="""Limpia esta transcripcion quitando las muletillas, sin cambiar el sentido:

bueno o sea lo que pasa es que el pedido llego el viernes pero faltaban dos cajas y nadie nos aviso nada"""),
dict(cat="transcript", lang="it", sys=None, prompt="""Trasforma questa registrazione in una nota breve e ordinata:

allora domani porto io i documenti pero mancano le firme di due persone e quindi forse si slitta a lunedi"""),
dict(cat="transcript", lang="ja", sys=None, prompt="""次の書き起こしから言い淀みを取り除いて読みやすくしてください。内容は変えないでください。

えーと、あの、来週の納品なんですけど、まあ、木曜になりそうで、もともと水曜の予定だったんですけど"""),
dict(cat="transcript", lang="pt", sys=None, prompt="""Limpe esta transcricao, tirando as repeticoes e mantendo os factos:

entao pronto a entrega ficou para quinta porque porque o fornecedor so tem stock na quarta a tarde"""),
dict(cat="transcript", lang="nl", sys=None, prompt="""Zet dit gedicteerde bericht om in een nette notitie:

eh ja dus de monteur komt dinsdag tussen negen en twaalf en hij zei dat hij de onderdelen al bij zich heeft"""),
dict(cat="transcript", lang="ko", sys=None, prompt="""다음 받아쓰기에서 군더더기를 빼고 읽기 좋게 다듬어 주세요. 내용은 바꾸지 마세요.

그러니까 음 내일 회의는 세 시로 미뤄졌고요 자료는 오늘 밤까지 보내 주시면 됩니다"""),
dict(cat="transcript", lang="is", sys=None, prompt="""Settu greinarmerki i thetta upptokuhandrit an thess ad breyta ordunum:

vid hittumst vid bryggjuna klukkan atta ef vedrid versnar tha frestum vid thessu fram a sunnudag eg laet thig vita"""),
dict(cat="transcript", lang="zh", sys=None, prompt="""请把下面这段口述整理成通顺的短段落，不要增加内容：

那个 明天的会改到三点了 资料我今天晚上发给你 对了 还要带上上次的报价单"""),

# --- short one-liners, greetings, follow-ups ---
dict(cat="short_chat", lang="en", sys=None, prompt="""Morning! Anything I should know before I start?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Hi there."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Thanks, that is exactly what I needed."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Can you do that again but shorter?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Go on."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Actually, scrap the last bit."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""What was the second option again?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Same again but for Norway."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Hmm, not quite. Try once more with a lighter touch."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Give me three more like that."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Can you explain that first line a bit more?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Perfect. Now in bullet points."""),
dict(cat="short_chat", lang="en", sys="You are terse by default and expand only when asked.", prompt="""What time is sunset in Reykjavik in late October, roughly?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Never mind, I worked it out."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""What do you think?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""One more question and then I will leave you alone: is it worth defrosting the freezer in winter?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Good evening. Long day. Tell me something cheerful."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Sorry, ignore that, wrong window."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Keep the tone but drop the last paragraph."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Yes please, the longer version."""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Hey, quick one: what is a decent free tool for cropping a PDF?"""),
dict(cat="short_chat", lang="en", sys=None, prompt="""Still there?"""),
dict(cat="short_chat", lang="de", sys=None, prompt="""Guten Morgen, kannst du das bitte noch kuerzer fassen?"""),
dict(cat="short_chat", lang="fr", sys=None, prompt="""Merci, c'est parfait. Une derniere chose: tu peux le mettre au passe?"""),
dict(cat="short_chat", lang="es", sys=None, prompt="""Hola, una pregunta rapida: como se llama la letra n con la rayita?"""),
dict(cat="short_chat", lang="it", sys=None, prompt="""Va bene cosi, grazie. Puoi aggiungere una riga finale?"""),
dict(cat="short_chat", lang="ja", sys=None, prompt="""おはようございます。もう少し短くしてもらえますか。"""),
dict(cat="short_chat", lang="pt", sys=None, prompt="""Boa tarde. Pode repetir isso de forma mais simples?"""),
dict(cat="short_chat", lang="nl", sys=None, prompt="""Dank je. Kun je er nog een voorbeeld bij zetten?"""),
dict(cat="short_chat", lang="is", sys=None, prompt="""Saell. Geturdu haft thetta adeins styttra?"""),
]
