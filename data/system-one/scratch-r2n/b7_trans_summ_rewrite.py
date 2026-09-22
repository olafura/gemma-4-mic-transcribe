# General: translation (35), summarisation of pasted text (40), rewriting / tone (35).
ROWS = [
# --- translation ---
dict(cat="translation", lang="en", sys=None, prompt="""Translate into Dutch, keeping it businesslike: 'Thank you for the revised quotation. We will confirm by Friday once the board has met.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Put this into Japanese as a polite customer email: 'Unfortunately the part is delayed until the 12th. We are happy to ship the rest now at no extra cost.'"""),
dict(cat="translation", lang="en", sys="You translate and then add one short note on anything culturally awkward.", prompt="""Into Korean, for a sign in an office kitchen: 'Please wash your own cup. The dishwasher runs at 5 pm.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate into Italian for a holiday rental listing: 'Two bedrooms, sea view from the kitchen, ten minutes on foot from the ferry.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""How would you render 'thoka' in English? An Icelandic friend used it about the weather and I am not sure of the register."""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate into Brazilian Portuguese, informal: 'We are running twenty minutes late, start without us and save us some bread.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Into Mandarin for a product label: 'Store in a cool dry place. Best before the date printed on the cap.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate this into German twice, once for a colleague you know well and once for a formal letter: 'Could you send the figures before the meeting?'"""),
dict(cat="translation", lang="en", sys=None, prompt="""What is a natural French equivalent of 'let's park that for now' in a meeting? Word for word sounds odd."""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate into Spanish for a museum caption, about twenty words: 'The boat was built in 1904 and fished the northern grounds for sixty years.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""I need this in Icelandic for a notice board: 'The lift will be out of service on Tuesday between nine and three.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate into Dutch and tell me whether 'gezellig' would fit anywhere: 'Come by after work, we will have soup and nobody has to stay long.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Render this legal sentence into plain Portuguese: 'The lessee shall be responsible for any damage beyond fair wear and tear.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate into Japanese and keep the joke if you can: 'Our deadline is Friday, which is a Tuesday in project time.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Into Italian, for a wedding invitation: 'We would love you to be there, but please, no speeches longer than the meal.'"""),
dict(cat="translation", lang="de", sys=None, prompt="""Uebersetze ins Englische, sachlich: 'Die Lieferung verzoegert sich um zwei Wochen, da der Zulieferer die Werkstoffpruefung wiederholen muss.'"""),
dict(cat="translation", lang="de", sys=None, prompt="""Wie sagt man 'Feierabend' auf Englisch, ohne dass es nach Arbeitsende im Sinne der Uhrzeit klingt?"""),
dict(cat="translation", lang="fr", sys=None, prompt="""Traduisez en anglais pour un site touristique: 'Le sentier longe la falaise sur trois kilometres, puis redescend vers le port.'"""),
dict(cat="translation", lang="fr", sys=None, prompt="""Comment traduire 'depaysement' en anglais sans perdre la nuance? Donnez deux options selon le contexte."""),
dict(cat="translation", lang="es", sys=None, prompt="""Traduce al ingles manteniendo el tono comercial: 'Adjuntamos el presupuesto actualizado; los precios se mantienen hasta fin de mes.'"""),
dict(cat="translation", lang="es", sys=None, prompt="""Como se dice 'sobremesa' en ingles? Necesito explicarselo a unos colegas que no conocen la costumbre."""),
dict(cat="translation", lang="it", sys=None, prompt="""Traduci in inglese per un menu, senza esagerare con gli aggettivi: 'tagliatelle al ragu di cinghiale, cotto lentamente nel vino rosso'."""),
dict(cat="translation", lang="ja", sys=None, prompt="""次の社内メールを自然な英語にしてください。「本件、先方の回答待ちのため、来週まで保留とさせてください。」"""),
dict(cat="translation", lang="ja", sys=None, prompt="""「お疲れ様です」を英語のメールの書き出しにするなら、どう書くのが自然ですか。いくつか例をください。"""),
dict(cat="translation", lang="pt", sys=None, prompt="""Traduza para ingles, mantendo o tom formal: 'Informamos que o prazo de entrega foi prorrogado por quinze dias uteis.'"""),
dict(cat="translation", lang="nl", sys=None, prompt="""Vertaal naar het Engels voor een handleiding: 'Draai de schroeven kruislings aan en controleer na tien minuten opnieuw.'"""),
dict(cat="translation", lang="ko", sys=None, prompt="""다음 문장을 자연스러운 영어로 번역해 주세요. '이번 주말까지 초안을 보내 주시면 월요일에 검토하겠습니다.'"""),
dict(cat="translation", lang="zh", sys=None, prompt="""请把这句话翻译成英文，语气正式一些：「因供应链延误，本批订单预计延后两周发货。」"""),
dict(cat="translation", lang="is", sys=None, prompt="""Hvernig thydir madur 'gluggavedur' a ensku? Eg tharf ad utskyra thetta fyrir vinnufelaga."""),
dict(cat="translation", lang="is", sys=None, prompt="""Thyddu thetta a ensku fyrir tolvupost til vidskiptavinar: 'Vidgerdin taf ist um tvo daga vegna vedurs, en vid sendum uppfaerda aaetlun a morgun.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate these three UI strings into German, keeping each under 20 characters: 'Save draft', 'Discard changes', 'Send for review'."""),
dict(cat="translation", lang="en", sys=None, prompt="""Give me a Korean version of this SMS, warm but brief: 'Your appointment is confirmed for Thursday at 14:00. Reply C to cancel.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate into French Canadian rather than European French, and note where they differ: 'Park in the lot behind the building and take the elevator to five.'"""),
dict(cat="translation", lang="en", sys=None, prompt="""I have a Spanish sentence I half understand: 'Nos pilló el toro con la entrega.' What does it actually mean and how idiomatic is it?"""),
dict(cat="translation", lang="en", sys=None, prompt="""Translate into Italian and Spanish both, same short sign: 'Quiet please, recording in progress.'"""),

# --- summarisation of a pasted paragraph ---
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise this in two sentences for someone who missed the meeting:

The harbour board agreed to postpone the dredging contract until the spring, mainly because the survey of the eastern basin came back later than expected and the figures in it did not match last year's soundings. Two members argued that waiting would push the work into the fishing season, but the chair noted that the contractor's price is fixed until June. The clerk was asked to write to the contractor confirming the delay and to circulate the survey to members before the next meeting, with the disputed measurements marked."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Give me a one paragraph plain language summary of this abstract:

Participants who slept six hours or less for five consecutive nights showed measurable declines in reaction time and working memory, with the largest effects in the final two days. Self-reported sleepiness, however, plateaued after the second night, so participants substantially underestimated their own impairment. A single night of recovery sleep restored reaction time but not working memory scores, and the authors note that the sample was small, drawn entirely from students, and tested in the afternoon, when circadian pressure is lowest. They argue that field studies of shift workers would be a better test of whether the working memory deficit persists beyond a week."""),
dict(cat="summarise", lang="en", sys="You summarise for busy readers. No preamble, no closing line.", prompt="""Condense this into three bullet points:

The supplier has confirmed that the panels will ship on the 14th rather than the 2nd. That lands after the installers are booked, and the installers charge half their fee if we cancel inside ten days. Marta suggests taking the earlier slot with the partial delivery and returning for the rest, which would add one day of labour but keep the building watertight before the weather turns. Petur would rather move the installers by a fortnight and accept the cancellation fee, on the grounds that two visits always cost more than the quote suggests. Nobody has asked the supplier whether a part shipment on the original date is possible at all."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise the argument of this passage in two sentences, and say what evidence it leans on:

Cities that removed an urban motorway did not see the traffic chaos opponents predicted. A share of the journeys disappeared entirely, some shifted to other times, and the rest spread across the remaining network. Traffic engineers call this disappearing traffic, and the pattern has now been documented in more than sixty schemes across Europe and North America. Critics point out that most of the studies measured the first two years only, and that the cities involved were already investing in public transport at the same time, which makes the effect of the road closure hard to isolate. The author accepts this but argues the direction of the finding has never been reversed."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Turn this into a three line status note for a chat channel:

Anna: the migration dry run finished at 04:10, about ninety minutes longer than we estimated. Bjorn: the slow part was the index rebuild on the events table, which we can do concurrently next time. Anna: agreed, but that changes the rollback plan, so we should walk through it on Thursday before we book the window. Bjorn: we also lost about twenty minutes to the backup verification, which ran twice because the first run did not report properly. Anna: put that on the list, but it is not on the critical path. Bjorn: fine, and I will ask support how much notice they need for the maintenance banner."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise this for a non-technical manager in under sixty words:

The incident began when a configuration change reduced the connection pool from fifty to five. Requests queued, health checks timed out, and the orchestrator restarted the pods, which made the queue worse. Service was restored by reverting the change; no data was lost, but around forty minutes of checkout traffic failed. The change had been reviewed and approved, but the reviewer read the diff as raising the pool rather than lowering it, and no alert existed on pool saturation. Two follow-up actions were agreed: an alert on saturated pools, and a rule that connection settings are changed only during office hours."""),
dict(cat="summarise", lang="en", sys=None, prompt="""What are the three main points here?

The report finds that remote hearings saved participants an average of four hours in travel and reduced non-attendance by a fifth. Judges, however, reported more difficulty assessing witness credibility, and litigants without a lawyer were markedly less likely to speak up. The authors recommend keeping remote hearings for procedural matters while returning contested evidence to the courtroom. They also note that the savings were unevenly distributed: professional users gained the most, while people joining from a phone on a poor connection often had to repeat themselves and dropped out of hearings more often. A small survey of court staff found broad support for keeping the remote option for short listings."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise this recipe intro into two sentences, keeping the practical bits:

This dough is wetter than most, around seventy-five percent hydration, which is what gives the open crumb. It needs a long cold ferment, at least sixteen hours in the fridge, and you should handle it as little as possible after the first fold. If your kitchen is warm, cut the bulk rise by an hour rather than adding flour."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Give me the gist of this in one sentence, then list anything that sounds like a commitment:

Thanks for the call. As discussed, we will look at moving the two smaller sites onto the new plan in January, assuming the migration tooling is ready, and I will send the updated pricing this week. We are not promising the API access before the second quarter."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise the disagreement in this thread neutrally:

Petur: we should cap uploads at 50 MB, anything larger is almost always a mistake. Hulda: half our architecture customers send drawings over 100 MB and they would hit that every day. Petur: then let us raise it per plan rather than for everyone. Hulda: that puts the limit in billing code, which is where limits go to die. Petur: then a flag on the account, set by support, with fifty as the default. Hulda: that is the same thing with extra steps, and support will set it to a thousand for anyone who complains."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Reduce this to a headline and a two sentence standfirst:

The council has approved a trial that closes the street beside the primary school to cars between half past eight and nine in the morning. Residents keep access, and the trial runs for six months with traffic counts before and after. A similar scheme in the next district cut the morning peak by a third."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise this voicemail transcript as a short note I can act on:

Hi, it's Gunnar from the garage. The car passed everything except the rear brake pads, which are close to the limit. We can do them Thursday morning if you leave it overnight Wednesday, otherwise it will be the week after. Give me a ring either way."""),
dict(cat="summarise", lang="en", sys=None, prompt="""In two sentences, what does this policy change actually require of employees?

From the first of next month, any purchase over 50,000 needs a second approver from outside your own team, recorded in the tool rather than by email. Purchases under that threshold are unchanged. Recurring subscriptions count as a single purchase at their annual value, which catches a lot of small monthly tools."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise the method and the finding separately:

The team compared two prompts on the same four hundred customer emails, scoring each reply against a rubric by two independent raters. The longer prompt improved rubric scores by six points on average, but the gain came almost entirely from the twenty percent of emails with more than one question in them."""),
dict(cat="summarise", lang="de", sys=None, prompt="""Fasse diesen Text in drei Saetzen zusammen:

Die Stadt will die Buslinie 12 ab Januar im Zehnminutentakt fahren lassen. Finanziert wird das zunaechst fuer ein Jahr aus Mitteln, die fuer den verschobenen Radwegbau vorgesehen waren. Kritiker im Rat halten das fuer eine Verschiebung des Problems, weil die Buslinie ohne eigene Spur weiterhin im Stau steht."""),
dict(cat="summarise", lang="fr", sys=None, prompt="""Resumez ce passage en deux phrases:

La bibliotheque ouvrira le dimanche a partir de septembre, avec une equipe reduite et sans service de pret entre midi et quatorze heures. Le budget vient d'une subvention regionale accordee pour deux ans. La direction precise que les horaires de semaine ne changeront pas."""),
dict(cat="summarise", lang="es", sys=None, prompt="""Resume esto en un parrafo corto y senala lo que queda sin decidir:

El ayuntamiento aprobo peatonalizar tres calles del centro durante seis meses a modo de prueba. Los comercios pidieron plazas de carga y descarga por la manana, y el acuerdo las incluye hasta las once. No se ha decidido que pasa con el aparcamiento de residentes al terminar la prueba."""),
dict(cat="summarise", lang="ja", sys=None, prompt="""次の文章を三行で要約してください。

市は来年度から古紙の回収を週二回に増やす方針を決めた。費用は年間およそ四千万円増える見込みで、財源は一般会計から充てる。分別の区分は変えないが、集積所の場所を一部見直すため、自治会への説明会を秋に開く。"""),
dict(cat="summarise", lang="pt", sys=None, prompt="""Resuma este texto em duas frases:

A empresa vai manter o trabalho hibrido, com dois dias obrigatorios no escritorio a partir de marco. As equipas que atendem clientes ficam de fora da regra e mantem o horario atual. A direcao diz que vai rever a medida ao fim de seis meses."""),
dict(cat="summarise", lang="nl", sys=None, prompt="""Vat dit samen in drie zinnen:

De gemeente vervangt de verlichting in het park door armaturen die na middernacht dimmen. Bewoners vroegen om meer licht bij de speeltuin, en dat deel blijft ongewijzigd. De kosten worden terugverdiend in ongeveer zeven jaar."""),
dict(cat="summarise", lang="it", sys=None, prompt="""Riassumi in due frasi e indica cosa resta da decidere:

Il consiglio ha approvato il rifacimento della piazza, ma i fondi coprono solo la prima fase. La seconda fase, con gli alberi e le panchine, dipende da un bando regionale che chiude a marzo."""),
dict(cat="summarise", lang="is", sys=None, prompt="""Dragdu thetta saman i tvaer setningar:

Sveitarfelagid aetlar ad fjolga leikskolaplassum um fjorutiu naesta haust med thvi ad taka i notkun husnaedi sem adur var nytt undir skrifstofur. Breytingin kostar um attatiu milljonir og krefst leyfis fra byggingarfulltrua. Bidlistinn er nu um hundrad born."""),
dict(cat="summarise", lang="ko", sys=None, prompt="""다음 글을 세 문장으로 요약해 주세요.

시는 내년부터 노후 상수도관 교체 사업을 시작한다. 예산은 삼 년에 걸쳐 나눠 집행되며, 공사 구간은 야간에 진행해 교통 혼잡을 줄일 계획이다. 다만 일부 구간은 주간 통제가 불가피하다고 밝혔다."""),
dict(cat="summarise", lang="zh", sys=None, prompt="""请用两句话概括下面这段话：

公司决定从下季度起把客服工单系统迁移到新平台，旧系统在迁移完成后再保留三个月只读。培训安排在迁移前两周，但夜班同事的场次还没定。"""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise the customer's problem and what they want, in one sentence each:

I ordered the desk on the 3rd, it arrived on the 11th with a cracked leg, and the replacement leg you sent is the wrong colour. I have now been waiting three weeks with a desk I cannot use. I do not want another part, I want the whole thing collected and refunded."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Pull out the decisions and the open questions from these notes:

Agreed to move the release to the 19th. Still unsure whether the feature flag defaults on for new accounts. Marketing wants a blog post; nobody has volunteered. Support asked for a one page summary by the 15th, which Petur said he would draft."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Give me a two sentence summary suitable for a release note:

The new version rewrites the importer so that a failed row no longer aborts the whole file. Failures are collected and written to a report alongside the successful rows. Files that previously took three passes now typically import in one."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise this in plain English for a tenant:

Under the terms of the lease the tenant is responsible for internal decoration and for any damage beyond fair wear and tear, while the landlord retains responsibility for the structure, the roof, the external walls and the shared drainage."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Three sentences, keep the numbers:

Attendance at the Saturday sessions rose from 18 to 41 over the term, mostly among families who live within a kilometre. The Tuesday evening session stayed flat at around a dozen. The coach thinks the difference is the pitch floodlights, which only cover the Saturday slot properly."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise what this reviewer liked and disliked, without quoting them:

The seats are comfortable and the boot is bigger than it looks, but the infotainment is a mess of nested menus and you cannot change the temperature without taking your eyes off the road. Fuel economy matched the claimed figure, which is rare."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Make this into a two sentence abstract for a programme:

The talk describes how a small team replaced a nightly batch job with a streaming pipeline over eight months, what broke on the way, and why they kept the batch job running in parallel for the first three of those months."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise the practical advice here as a checklist:

Before a long drive, check tyre pressures cold, top up the screenwash, and look at the tread on the spare as well as the four fitted. Plan a stop every two hours even if you do not feel tired, and keep the fuel above a quarter in winter so condensation does not build in the tank."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Shorten this to about forty words without losing the caveat:

Early results suggest the new onboarding flow improves week-one retention by around four points, but the test ran during a promotion, so some of that lift may be the discount rather than the flow. We plan to rerun it in a quiet week."""),
dict(cat="summarise", lang="en", sys=None, prompt="""What is this email really asking for? Summarise in one line and flag the deadline:

Following our conversation, and assuming nothing changes on your side, we would appreciate confirmation of the revised scope in writing before the end of the week so that the subcontractor can hold the crane booking for the 22nd."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise this in three sentences and keep it neutral:

The museum will charge for the special exhibitions from April while keeping the permanent collection free. The trustees say ticketing pays for touring shows that would otherwise not come. A staff representative noted that the last time charges were introduced, visits fell for two years before recovering."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Condense these minutes to five lines:

The committee reviewed the pier repairs. Divers found more corrosion on the western piles than the survey suggested. The engineer will re-price the work with a cathodic protection option. Members asked whether the pier can stay open during the work; the engineer thought partly. A decision was deferred to the next meeting, with a written report requested a week in advance."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise the two options being weighed and what each costs:

We can either extend the current contract by a year at a six percent uplift, or go to tender now and probably land ten percent lower but spend roughly two months of someone's time running it. The incumbent knows the estate, which saved us during the flood last winter."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Give me the one thing a reader needs to remember from this:

Most people think the danger of a small leak is the water. In a flat roof it is the insulation: once it is wet it stays wet, loses most of its value, and the repair becomes a strip and rebuild rather than a patch."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Summarise this for the top of a report, formally:

Over the year the service handled 14,200 calls, up nine percent, with the average wait falling from four minutes to two and a half after the new rota began in May. Abandoned calls halved. Staff turnover, however, rose to nineteen percent, the highest in five years, and exit interviews point at the rota itself rather than pay. Training days fell from six to two per person as cover was tightened. The service met its target on first-contact resolution for the first time, though the definition changed in March, so the two halves of the year are not strictly comparable."""),
dict(cat="summarise", lang="en", sys=None, prompt="""Two sentences, aimed at a parent:

The school trip leaves at seven on the 14th and returns around eight that evening. Children need a packed lunch, a waterproof and sturdy shoes; the coach has no toilet, and there is one stop each way."""),

# --- rewriting / tone ---
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite this so it sounds firm but not aggressive: 'As I have already said twice, the invoice was due on the 3rd and I am still waiting.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Make this shorter and less hedged: 'I was just wondering whether it might perhaps be possible to maybe move our meeting, if that is not too inconvenient for you.'"""),
dict(cat="rewrite", lang="en", sys="You edit business writing. Cut words, keep meaning, never add new claims.", prompt="""Tighten this: 'It is important to note that our team has, over the course of the last several months, been actively working towards the goal of improving the overall reliability of the platform.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite this rejection so it is kind but does not invite a debate: 'We went with someone else. Your experience did not really fit what we needed.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Turn this into something a twelve year old could read without losing the meaning: 'Sustained exposure to elevated noise levels is associated with cumulative auditory deterioration.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite for a text message, warm, under twenty words: 'I regret to inform you that I will be unable to attend the dinner on Saturday due to a prior engagement.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Make this sound like a person rather than a policy: 'Refund requests submitted outside the stated window cannot be accommodated as per company policy.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite in the active voice and say which changes you made: 'It was decided by the committee that the report would be delayed until such time as the figures are verified.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Give me three versions of this subject line, one neutral, one urgent, one friendly: 'Contract renewal'."""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite this bio in the first person and make it less boastful: 'Kristin is a visionary leader with a proven track record of driving transformative outcomes across multiple verticals.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Soften this without becoming vague: 'Your code review comments were unhelpful and mostly about style.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite this apology so it takes responsibility without grovelling: 'Sorry, sorry, this is entirely my fault, I completely messed it up, it will never happen again, I feel awful.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Turn these notes into two polished sentences: 'window seals gone / draught in bedroom / quoted 180k / can do in march / need decision by fri'."""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Make this announcement less corporate: 'We are excited to announce that we will be sunsetting the legacy dashboard in order to better serve our users going forward.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite as a one sentence commit message in the imperative: 'I changed the retry code so it waits longer between attempts and stops after five tries instead of ten.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Same content, but for a formal letter to a bank rather than an email to a friend: 'Hi, the standing order came out twice this month, can you sort it?'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite this so it does not assume the reader is at fault: 'You forgot to attach the file again.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Make it shorter."""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Can you make that last version a bit warmer, and keep the second sentence as it is?"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite the middle paragraph only, the rest was fine."""),
dict(cat="rewrite", lang="en", sys=None, prompt="""That is close. Drop the exclamation marks and make the ending less abrupt."""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite this listing to be honest about the flaws without putting people off: 'Charming cottage, needs some modernisation, original windows throughout, garden a little overgrown.'"""),
dict(cat="rewrite", lang="en", sys="You are an editor for a plain-language public sector style guide.", prompt="""Rewrite this for a council leaflet: 'Residents are advised that refuse collection schedules will be subject to temporary amendment during the festive period.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Turn this into a standup update of two lines: 'Yesterday I mostly fought with the CI cache, got it working by the afternoon, then started on the importer but only got as far as reading the spec.'"""),
dict(cat="rewrite", lang="en", sys=None, prompt="""Rewrite this so it works read aloud, with shorter sentences: 'Given the constraints outlined above, and notwithstanding the delay, we believe the proposed approach remains viable.'"""),
dict(cat="rewrite", lang="de", sys=None, prompt="""Formuliere diese Mail hoeflicher, ohne dass sie unterwuerfig klingt: 'Ich habe letzte Woche gefragt und noch keine Antwort bekommen. Kann das heute erledigt werden?'"""),
dict(cat="rewrite", lang="de", sys=None, prompt="""Kuerze diesen Absatz auf die Haelfte und behalte die Zahlen: 'Wir konnten die Bearbeitungszeit von durchschnittlich zwoelf auf sieben Tage senken, unter anderem durch eine neue Zustaendigkeitsregelung im Team.'"""),
dict(cat="rewrite", lang="fr", sys=None, prompt="""Reformule ce message pour qu'il soit plus direct sans etre sec: 'Je me permets de revenir vers vous concernant le dossier dont nous avions parle il y a quelque temps deja.'"""),
dict(cat="rewrite", lang="es", sys=None, prompt="""Reescribe esto para un cartel, con frases cortas: 'Se informa a los usuarios de que el servicio de prestamo permanecera suspendido durante las obras de acondicionamiento.'"""),
dict(cat="rewrite", lang="it", sys=None, prompt="""Rendi questo messaggio piu cordiale senza allungarlo: 'Il pagamento risulta ancora non effettuato. Si prega di provvedere.'"""),
dict(cat="rewrite", lang="ja", sys=None, prompt="""次の文章を、社外向けに丁寧な言い方へ書き直してください。「納期は来週になります。今回は間に合いませんでした。」"""),
dict(cat="rewrite", lang="pt", sys=None, prompt="""Reescreva este aviso em linguagem simples: 'Comunica-se aos condominos que a manutencao do elevador implicara interrupcao do servico.'"""),
dict(cat="rewrite", lang="nl", sys=None, prompt="""Herschrijf dit zodat het vriendelijker klinkt maar even duidelijk blijft: 'Je hebt de deadline gemist en dat kost ons tijd.'"""),
dict(cat="rewrite", lang="ko", sys=None, prompt="""다음 문장을 더 간결하고 자연스럽게 다듬어 주세요. '이번 건에 대해서는 내부적으로 검토를 진행하고 있는 상황이라고 말씀드릴 수 있겠습니다.'"""),
dict(cat="rewrite", lang="is", sys=None, prompt="""Umskrifadu thetta svo thad hljomi hlylegar en jafn skyrt: 'Thu skilaðir ekki gognunum a rettum tima og thad tafdi verkefnid.'"""),
]
