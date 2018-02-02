/*
 *	SHAPES.JS
 */


//	### Routine di uso generale ###

//	calcola e restituisce il quadrato della distanza 
//	tra i due punti a e b della variabile in ingresso
function delta2(item) {
    var dx = item.b.x - item.a.x,
        dy = item.b.y - item.a.y;
    return dx * dx + dy * dy;
}

//	calcola e restituisce il quadrato della distanza 
//	tra la variabile point e le coordinate x e y
function dist2(point, x, y) {
    var dx = x - point.x,
        dy = y - point.y;
    return dx * dx + dy * dy;
}

//	calcola e restituisce il quadrato della distanza 
//	dall'estremo più lontano alle coordinate x e y
function maxDist2(item, x, y) {
    return Math.max(dist2(item.a, x, y), dist2(item.b, x, y))
}

//	calcola e restituisce l'estremo più vicino alle coordinate x e y
function nearest(item, x, y) {
    return (dist2(item.a, x, y) < dist2(item.b, x, y)) ? item.a : item.b
}

//	calcola e restituisce il centro del quarto di circonferenza
function arcCenter(item) {
    var c = [],
        sumX = item.b.x + item.a.x,
        difX = item.b.x - item.a.x,
        sumY = item.b.y + item.a.y,
        difY = item.b.y - item.a.y,
        verso = (item.shape == "arc (ccw)") ? +1 : -1;
    c.x = (sumX + verso * difY) / 2;
    c.y = (sumY - verso * difX) / 2;
    return c;
}

//	restituisce la descrizione testuale della linea per il path dello SVG
function toLine(item) {
    return ["M", item.a.x, item.a.y, "L", item.b.x, item.b.y].join(" ");
}

//	restituisce la descrizione testuale dell'arco per il path dello SVG 
//	i punti a e b di item rappresentano gli estremi del quarto d'arco di 
//	circonferenza da tracciare in senso orario o anti-orario, 
//	a seconda del tipo di shape
function toArc(item) {
    return ["M", item.a.x, item.a.y,
        "A", item.radius, item.radius, 0, 0, item.verso, item.b.x, item.b.y].join(" ");
}

//      genera l'espressione formale che descrive l'insieme 
//      degli stroke che costutiuscono la gesture complessiva
function printExpr() {
    const unit = 20;    //  unità arbitraria che determina la scala della rappresentazione (e la conseguente approssimazione)
//  variabili temporanee di appoggio per le coordinate intermedie
    var item = null,
        from = [],
        tmp = "", cw = "",
        x, y, dx, dy;

    for (var i = 0; i < gestures.length; i++) {
        item = gestures[i];
        //  coordinate dell'estremo iniziale convertite di un'unità di misura
        x = Math.round(item.a.x / unit);
        y = Math.round(item.a.y / unit);

        //  se è la prima shape
        if (i == 0)
        //  per convenzione, la prima shape deve partire dall'origine (0,0)
            dx = dy = 0;
        else {
            //  altrimenti calcolo le coordinate relative, ossia lo spostamento rispetto a from
            dx = x - from.x;
            dy = y - from.y;
        }
        //  l'estremo iniziale diventa il prossimo from
        from.x = x;
        from.y = y;
        //	se le differenze non sono nulle, allora l'estremo
        //	iniziale non coincide e quindi genero una nuova
        //	origine P non concatenata (quindi senza il simbolo +)
        if ((i == 0)
            || (item.shape == "point")
            || (dx || dy))
            tmp += " P(" + dx + "," + dy + ")";
        //  determinazione del codice identificativo della shape
        switch (item.shape) {
            case "line":
                tmp += " + L"
                break;
            case "arc (cw)":
                tmp += " + A"
                cw = ", true"
                break;
            case "arc (ccw)":
                tmp += " + A"
                cw = ", false"
                break;
        }
        //  a meno che non si tratti di un punto, calcolo le coordinate
        //  dello spostamento e genero l'espressione corrispondente
        if (item.shape != "point") {
            x = Math.round(item.b.x / unit);
            y = Math.round(item.b.y / unit);
            dx = x - from.x;
            dy = y - from.y;
            from.x = x;
            from.y = y;
            tmp += "(" + dx + "," + (- dy) + cw +")";
        }
    }
    //  infine, stampo l'espressione
    d3.select('#printer').text(tmp);
}


//	rende attiva la shape "item" 
//	disattiva quella precedentemente attiva, se esiste
//	aggiorna lo status
//	e restituisce 1 in caso di successo
function select(item = null) {
    if (gestures.length == 0)
        active = null;
    if ((item === null) && (gestures.length > 0))
        item = gestures[gestures.length - 1];

//	se item coincide con la shape correntemente attiva restituisco 0
    if (item === active) return 0;

//	altrimenti rimetto il colore normale alla shape precendente attiva
    if (active !== null)
        active.style("stroke", shapeColor);
//	rendo attivo il nuovo selezionato
    active = item;
//	gli assegno il colore apposito
    active.style("stroke", activeColor);
//	quindi aggiorno la riga di stato
    updateStatus();
//	e infine restituisco 1 al chiamante
    return 1;
}

//	(ri)traccia le varie shape chiamando le opportune funzioni 
//	per generare le corrispondenti descrizioni testuali SVG
function redraw(item) {
    switch (item.shape) {
        case "point":
        case "line":
            return item.attr("d", toLine(item));
        //	break;	// superfluo
        case "arc (cw)":
        case "arc (ccw)":
            //	precalcolo il raggio e le coordinate del centro anche per altri usi (selezione)
            item.center = arcCenter(item);
            item.radius = Math.sqrt(delta2(item) / 2.0);
            //	essendo l'arco un quarto di circonferenza, il raggio è pari alla
            //	lunghezza della corda diviso radice di due, quindi per semplicità
            //	divido direttamente per due delta2(item), che è il quadrato della
            //	lunghezza della corda, PRIMA di estrarne la radice quadrata
            //	ossia sqrt(d*d) / strt(2) == sqrt(d*d/2)
            item.verso = (item.shape == "arc (ccw)") ? 0 : 1;
            return item.attr('d', toArc(item));
        //	break;	// superfluo
    }
}

//	*** PROVVISORIO *** Gestione della RIGA DI STATO
//	( testo descrittivo dello stato corrente del sistema )
function updateStatus() {
    const defaultStatus = "Unistroke mod";
    //	se non è attiva nessuna modalità, mostro il testo predefinito
    if (mode == "")
        d3.select('#statusline').text(defaultStatus);
    //	altrimenti stampo la modalità e il tipo di shape correnti
    else if (mode == "draw")
        d3.select('#statusline').text("Draw " + shapeType);
    else
        d3.select('#statusline').text(mode + " " + active.shape);
}

/*
	### Variabili Globali ###

	gestures - vettore che contiene gli SVG generati nel DOM
	element - vettore che contiene i punti della shape usato solo per il log
	shapeType - tipo della shape (linea, punto, arco)
	active - riferimento alla shape correntemente selezionata
	uniStroke - 0/1, forza l'operatività in unistroke
	bgColor, shapeColor, activeColor - colori di background, shape e selezione attiva
	modes - vettore che elenca le varie modalità operative
	mode - modalità corrente
*/

var bgColor = '#fff', shapeColor = '#557', activeColor = '#aaf',
    gestures = [],
    active = null,
    shapeType = "",
    uniStroke = true,
    modes = ["draw", "edit", "move"],
    mode = "",

//	imposto adesso il gestore degli eventi di trascinamento del mouse 
//	(localizzato sullo SVG con ID univoco #main, che contiene la griglia 
//	di fondo), generata direttamente nella dichiarazione dello stesso 
//	SVG nell'html - salvo inoltre il riferimento restituito, in quanto 
//	mi servirà per inserirvi le shape generate 
    svg = d3.select("#main")
        .call(d3.drag()	//	avvia i due gestori degli eventi di trascinamento
            .on("start", dragstarted)
            .on("end", dragended)
        );

/*
	La funzione "dragended" viene chiamata alla fine del tracciamento
	e si occupa di verificare se il nuovo elemento generato debba essere
	eliminato in quanto nullo, oppure accettato e in tal caso anche 
	conservato nell'apposito stack delle shapes (inoltre le informazioni 
	della shape sono inserite nel log del browser per comodità di debug)
 */
function dragended() {
    switch (mode) {
        case "edit":
        case "move":
//	attualmente nelle modalità diverse da "draw" non vi è nulla da eseguire
            break;
        case "draw":
            var element = [];	// contiene le informazioni sulla shape per il log
            element.shape = active.shape;
            if (active.shape == "point") {
                //	salvo le coordinate dell'unico punto della shape
                element.x = active.a.x;
                element.y = active.a.y;
            } else {
                //	se la shape risulta essere di dimensione nulla, viene eliminata
                if (delta2(active) == 0) {
                    active.remove();
                    //	se nel vettore sono presenti altre shape, rimetto attiva l'ultima
                    if (gestures.length > 0) {
                        active = gestures[gestures.length - 1];
                        active.style("stroke", activeColor);
                    }
                    return;
                }
                //	salvo le coordinate di entrambi i punti della shape
                element.xa = active.a.x;
                element.ya = active.a.y;
                element.xb = active.b.x;
                element.yb = active.b.y;
            }

//	salvo il riferimento alla nuova shape in coda nell'apposito vettore
            gestures.push(active);
//	ed effettuo il log nella console
            console.log(element);

            break;
    }
    printExpr();
}


/*
	La funzione checkSelection viene chiamata all'inizio delle 
	modalità "edit" e "move", per verificare se il punto iniziale
	del click comporta un cambiamento della shape attiva, nel qual 
	caso restituisce 1 altrimenti 0
 */
function checkSelection(x, y) {
    if (gestures.length > 1) {
        var item = [], selected = -1;
        for (var i = 0; i < gestures.length; i++) {
            item = gestures[i];	// shape da testare
            //	procedo adesso a testare la distanza del click dalla shape
            //	con tolleranza calcolata per tentativi ed appropriata ad ogni caso
            switch (item.shape) {
                case "point":
                    //	distanza di x e y dalla shape
                    //	(si può testare indifferentemente uno dei due estremi, che coincidono)
                    if (Math.abs(Math.sqrt(dist2(item.a, x, y)) < 3))
                        selected = i;
                    break;
                case "line":
                    //	se il click è sulla linea o nelle immediate vicinanze,
                    //	allora la somma delle distanze dai due estremi sarà di
                    //	poco superiore alla distanza tra gli estremi stessi
                    if (Math.abs(
                            Math.sqrt(dist2(item.a, x, y))
                            + Math.sqrt(dist2(item.b, x, y))
                            - Math.sqrt(delta2(item)))
                        < 0.05)
                        selected = i;
                    break;
                case "arc (cw)":
                case "arc (ccw)":
                    //	se il click è sull'arco o nelle immediate vicinanze, allora
                    //	la distanza dal centro sarà di poco differente dal raggio stesso
                    //	ma inoltre, verifico anche che la massima distanza dai due estremi
                    //	sia inferiore alla lunghezza della corda, per escludere che il punto
                    //	si trovi all'esterno dell'arco, anche se pur sempre sulla circonferenza
                    if ((Math.abs(Math.sqrt(dist2(item.center, x, y)) - item.radius) < 3)
                        && (maxDist2(item, x, y) <= delta2(item)))
                        selected = i;
                    break;
            }
            //	se è stato selezionato un elemento, il ciclo viene interrotto
            if (selected > -1)
                break;
        }

        //	se è stato selezionato un elemento
        //	ed è differente da quello correntemente attivo
        //	restituisco 1
        if ((selected > -1) && select(gestures[selected]))
            return 1;
    }	// in tutti gli altri casi restituisco 0
    return 0;
}


/*
	La funzione "dragstarted" imposta il gestore degli eventi per ognuna delle 
	modalità operative, che sono attualmente tre: "draw", "move" e "edit"
*/

function dragstarted() {
    var point, twinA, twinB, minA, minB, tmp, last;
    switch (mode) {
        case "move":
            //	se non è presente alcuna shape, oppure se è cambiata la shape selezionata
            //	allora l'eventuale spostamento del mouse viene ignorato
            if ((gestures.length < 1) || checkSelection(d3.event.x, d3.event.y)) return;

            if (uniStroke)
                d3.event.on("drag", function () {	// gestore dell'evento di trascinamento
                    //	in unistroke sposta tutte le shape della stessa entità
                    //	del movimento del mouse e quindi le ridisegna tutte
                    for (var i = 0; i < gestures.length; i++) {
                        gestures[i].a.x += d3.event.dx;
                        gestures[i].a.y += d3.event.dy;
                        gestures[i].b.x += d3.event.dx;
                        gestures[i].b.y += d3.event.dy;
                        redraw(gestures[i]);
                    }
                });
            else
                d3.event.on("drag", function () {	// gestore dell'evento di trascinamento
                    //	sposta entrambi i punti della shape della stessa entità
                    //	del movimento del mouse e quindi la ridisegna
                    active.a.x += d3.event.dx;
                    active.a.y += d3.event.dy;
                    active.b.x += d3.event.dx;
                    active.b.y += d3.event.dy;
                    redraw(active);
                });
            break;

        case "edit":
            //	se non è presente alcuna shape
            //	oppure se è cambiata la shape selezionata
            //	allora l'eventuale spostamento del mouse viene ignorato
            if ((gestures.length < 1) || checkSelection(d3.event.x, d3.event.y))
                return;

            if (!uniStroke) { // multistroke
                //	se sono in multistroke, devo spostare soltanto un estremo della
                // 	shape attiva, quello più vivino al click, che salvo in point
                point = nearest(active, d3.event.x, d3.event.y);
                d3.event.on("drag", function () {	// gestore dell'evento di trascinamento
                    //	aggiorno le coordinate del punto da trascinare
                    if (active.shape == "point") {
                        active.a.x = active.b.x = d3.event.x;
                        active.a.y = active.b.y = d3.event.y;
                    } else {
                        point.x = d3.event.x;
                        point.y = d3.event.y;
                    }
                    //	quindi ridisegno la shape
                    redraw(active);
                });
            } else { // unistroke
                //	altrimenti, se sono in unistroke, devo individuare e salvare
                //	i due estremi "gemelli" delle shape consecutive da spostare insieme

                //	testo ciclicamente tutte le shape per trovare gli estremi iniziale (a) e
                //	finale (b) più vicini al click e ne salvo i riferimenti in twinA e twinB
                //	nonché le rispettive distanze in minA e minB;
                //	le quattro variabili sono inizializzate con la prima shape, riferita
                //	in gestures[0], e il ciclo parte quindi dalla shape successiva
                minA = dist2(gestures[0].a, d3.event.x, d3.event.y);
                minB = dist2(gestures[0].b, d3.event.x, d3.event.y);
                twinA = twinB = gestures[0];
                for (var i = 1; i < gestures.length; i++) {
                    tmp = dist2(gestures[i].a, d3.event.x, d3.event.y);
                    if (tmp < minA) {
                        minA = tmp;
                        twinA = gestures[i];
                    }
                    tmp = dist2(gestures[i].b, d3.event.x, d3.event.y);
                    if (tmp < minB) {
                        minB = tmp;
                        twinB = gestures[i];
                    }
                }

                //	se i due estremi sopra calcolati non coincidono esattamente
                //	(secondo la tolleranza stabilita), allora sposto solo il più vicino
                if (Math.abs(minA - minB) > 10)
                    if (minA < minB) twinB = 0;
                    else twinA = 0;
                //	IMPORTANTE!
                //	il punto non può far parte di un unistroke, quindi
                //	devo limitarmi a spostare il solo punto anche nel caso
                //	che casualmente coincida con l'estremo di un altra shape
                else if (twinA.shape == "point") twinB = 0;
                else if (twinB.shape == "point") twinA = 0;

                d3.event.on("drag", function () {	// gestore dell'evento di trascinamento
                    if (twinA != 0) {	// se twinA è un riferimento non nullo,
                        //	sposto il primo estremo (a)
                        twinA.a.x = d3.event.x;
                        twinA.a.y = d3.event.y;
                        //	ma se è un punto devo spostare entrambi gli estremi
                        if (twinA.shape == "point") {
                            twinA.b.x = d3.event.x;
                            twinA.b.y = d3.event.y;
                        }
                        redraw(twinA);
                    }
                    if (twinB != 0) {	// se twinB è un riferimento non nullo,
                        //	sposto il secondo estremo (b)
                        twinB.b.x = d3.event.x;
                        twinB.b.y = d3.event.y;
                        //	ma se è un punto devo spostare entrambi gli estremi
                        if (twinB.shape == "point") {
                            twinB.a.x = d3.event.x;
                            twinB.a.y = d3.event.y;
                        }
                        redraw(twinB);
                    }
                });
            }
            break;

        case "draw":
            //	salvo un riferimento all'ultima shape
            if (gestures.length == 0)
                last = null;
            else
                last = gestures[gestures.length - 1];

            //	disattivo l'elemento attivo in precedenza, se presente
            if (active !== null)
                active.style("stroke", shapeColor);

            //	creo un elemento path con classe "shape", inserendolo all'interno
            //	dello SVG, e ne salvo il riferimento restituito in active
            active = svg.append("path").attr("class", "shape");
            //	definisco nell'elemento stesso i vari attributi:
            active.shape = shapeType;	//	il tipo di shape da tracciare
            if (active.shape != "point") {	// se non è un punto,
                //	imposto un marker a forma di freccia sul secondo estremo
                active.attr("marker-end", "url(#arrow)");
                //	e un marker a forma di punto sul primo estremo, se è una multistroke
                //	oppure se è la prima shape di un unistroke, oppure se segue un punto
                if (!uniStroke || (gestures.length == 0) || (last.shape == "point"))
                    active.attr("marker-start", "url(#dot)");
            }
            active.a = [];	//	il primo punto
            active.b = [];	//	il secondo punto
            if (uniStroke && (active.shape != "point")
                && (gestures.length > 0) && (last.shape != "point")) {
                // se in modalità unistroke e se non è un punto,
                // il primo estremo parte dal secondo dell'ultimo stroke,
                // se questo esiste e se non è un punto a sua volta
                active.a.x = last.b.x;
                active.a.y = last.b.y;
                active.b.x = d3.event.x;
                active.b.y = d3.event.y;
            } else {
                // in tutti gli altri casi, inizialmente i due punti sono uguali
                active.a.x = active.b.x = d3.event.x;
                active.a.y = active.b.y = d3.event.y;
            }
            //	quindi gli imposto il colore attivo e lo traccio per la prima volta
            redraw(active).style("stroke", activeColor);

            d3.event.on("drag", function () {	// gestore dell'evento di trascinamento
                //	aggiorno il secondo punto della shape
                active.b.x = d3.event.x;
                active.b.y = d3.event.y;
                //	nel solo caso in cui si tratti di un punto, allora
                //	anche il primo punto deve coincidere col secondo
                if (active.shape == "point") {
                    active.a.x = d3.event.x;
                    active.a.y = d3.event.y;
                }

                //	se la shape ha dimensione nulla e non è un punto allora
                //	la rendo non visibile attribuendole il colore di fondo
                if ((delta2(active) == 0) && (active.shape != "point"))
                    active.style("stroke", bgColor);
                else
                    active.style("stroke", activeColor);

                //	infine ritraccio la shape aggiornata
                redraw(active);
            });
            break;
    }
}


/*
	Il codice seguente rende attivi e gestisce i pulsanti della palette.

	In base all'indice del bottone, vengono definiti i parametri che saranno letti
	dal gestore dell'evento per stabilire come interpretare il movimento del mouse 
	e conseguentemente tradurlo nell'azione voluta.
	
	Il primo bottone attiva e disattiva l'operatività unistroke.

	I successivi quattro bottoni attivano la modalità "draw" e selezionano il 
	tracciamento della shape corrispondente: 
		punto, linea, arco orario e arco anti-orario.

	I successivi due, selezionano rispettivamente le modalità "move" e "edit".

	Infine, l'ultimo bottone svolge la funzione di eliminare l'elemento attivo e 
	contemporaneamente portando attivo l'elemento sottostante quello eliminato.



	### Variabili Globali ###

	palette - vettore delle voci di menu
	action - voce del menu correntemente attiva
*/

var palette = ["single", "point", "line", "arc (cw)", "arc (ccw)", "move", "edit", "del"],
    action = 0;

//	attivo il gestore dell'evento click su tutti gli elementi ".button" di SVG
d3.selectAll('.button').on('click', function (d, i) {
//	i è l'indice del cerchio cliccato
//	(nel codice lo imposto a -1 quando non si deve cambiare l'action corrente)
    var paletteIndex = Math.trunc(i/2);
    var currButton = d3.select(d3.selectAll('.button').nodes()[paletteIndex * 2]);
    var newAction = palette[paletteIndex];	//	stringa corrispondente al cerchio cliccato

    switch (newAction) {
        case "single":
            uniStroke = !uniStroke;  // inverto lo stato (on/off)
            // imposto il colore in funzione dello stato attivo
            currButton.style("fill", uniStroke ? "#def" : "#fed");
            break;
        case "point":
        case "line":
        case "arc (cw)":
        case "arc (ccw)":
            //	in caso di nuova shape da tracciare
            mode = "draw";	//	imposto la modalità a "draw"
            shapeType = newAction;	//	e salvo il tipo di shape richiesta
            select();  // se è presente almeno una shape rendo attiva l'ultima
            break;
        case "edit":
        case "move":
            //	in caso di modalità diversa da "draw"
            if (gestures.length > 0) //	se esiste almeno una shape
                mode = newAction;	// allora imposto la nuova modalità
            else	// altrimenti la ignoro e rimane la precedente
                i = -1;	// e anche l'action non deve essere cambiata
            break;
        case "del":
            i = -1;	//	l'action corrente rimane inalterata
            if (gestures.length > 0) {	//	se esiste almeno una shape
                gestures.splice(gestures.indexOf(active), 1);	//	rimuovo quella attiva
                active.remove();	//	e la elimino anche dallo SVG
                //  se è rimasta almeno un'altra shape la rendo attiva (select())
                if (!select() && (mode != "draw")) {
                    //	altrimenti preservo la modalità corrente di tracciamento, se impostata
                    mode = "";
                    action.style("fill", "#fed");
                }
            }
            break;
    }

    if (paletteIndex > 0) {	//	se l'action corrente deve essere cambiata
        if (action != 0)	// se ne esiste una precedente
            action.style("fill", "#fed");	// le riassegno il colore di fondo
        action = currButton;	// salvo il riferimento al cerchio corrente
        action.style("fill", "#def");	// e gli assegno il colore attivo
    }
    updateStatus();
    printExpr();
});

/*
*/
