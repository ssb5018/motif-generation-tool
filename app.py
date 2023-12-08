from flask import Flask, render_template, request, redirect, url_for, make_response
import analyse_input
import asyncio

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
async def generate():
    form = {'gcContentMinPercentage': 25, 
            'gcContentMaxPercentage': 65, 
            'maxHomopolymer': 5,
            'maxHairpin': 1,
            'loopMin': 6,
            'loopMax': 7,
            'payloadSize': 5, 
            'payloadNum': 10,
            'keySize': 2,
            'keyNum': 2,
            'homVisible': False,
            'hairpinVisible': False,
            'gcVisible': False,
            }

    if request.method == 'POST':

        if "analyseSeqSubmission" in request.form:
            # Form being submitted, get data from form.
            gcContentMinPercentage = 25
            gcContentMaxPercentage = 65
            max_homopolymer = 1
            maxHairpin = 1
            loopMin = 6
            loopMax = 7
            payloadSize = int(request.form['payloadSize'])
            payloadNum = int(request.form['payloadNum'])
            keySize = int(request.form['keySize'])
            keyNum = int(request.form['keyNum'])
            constraints = set()
            if request.form['homVisible'] == 'True':
                constraints.add('hom')
                max_homopolymer = int(request.form['maxHomopolymer'])
            if request.form['hairpinVisible'] == 'True':
                constraints.add('hairpin')
                maxHairpin = int(request.form['maxHairpin'])
                loopMin = int(request.form['loopMin'])
                loopMax = int(request.form['loopMax'])
            if request.form['gcVisible'] == 'True':
                constraints.add('gcContent')
                gcContentMinPercentage = float(request.form['gcContentMinPercentage'])
                gcContentMaxPercentage = float(request.form['gcContentMaxPercentage'])

            form = {'gcContentMinPercentage': gcContentMinPercentage, 
                    'gcContentMaxPercentage': gcContentMaxPercentage, 
                    'maxHomopolymer': max_homopolymer,
                    'maxHairpin': maxHairpin,
                    'loopMin': loopMin,
                    'loopMax': loopMax,
                    'payloadSize': payloadSize, 
                    'payloadNum': payloadNum,
                    'keySize': keySize,
                    'keyNum': keyNum,
                    'homVisible': request.form['homVisible'],
                    'hairpinVisible': request.form['hairpinVisible'],
                    'gcVisible': request.form['gcVisible'], 
                    }

            keys, payloads, motifs, isValid = await analyse_input.generate_keys_and_payloads(form,\
                                                                                    constraints)

            # Render the page
            return render_template('index.html', payloads=payloads, motifs=motifs, keys=keys, \
                                    form=form, isValid=isValid, submitted=True)
        
        return render_template('index.html', payloads="", keys="", motifs="", form=form, \
                                isValid=False, submitted=False)

    return render_template('index.html', payloads="", keys="", motifs="", form=form, \
                            isValid=False, submitted=False)


if __name__ == "__main__":
   app.run(debug=True)


