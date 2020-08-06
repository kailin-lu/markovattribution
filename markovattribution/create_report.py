from absl import app, flags

from datetime import datetime, timedelta
from google.cloud import bigquery

import numpy as np
import pandas as pd
from pandas.plotting import table

import matplotlib.pyplot as plt
from markovattribution.model import MarkovAttribution

from data import get_query_str

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch


flags.DEFINE_string('start_date', '2020-06-01', 'Start date of conversions, %Y-%m-%d')
flags.DEFINE_string('end_date', '2020-07-01', 'End date of conversions, %Y-%m-%d')
flags.DEFINE_string('report_name', 'attribution_report', 'File name of report, does not include .pdf')

FLAGS = flags.FLAGS


def download(start_date, end_date):
    """

    """
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    path_start_date = start_date_dt - timedelta(days=100)
    path_start = path_start_date.strftime('%Y-%m-%d')

    client = bigquery.Client()
    query_str = get_query_str(start_date, path_start, end_date)
    data = client.query(query_str).to_dataframe()

    multitouch = data[data['touchpoints'] > 1]
    multitouch['majority_retention'] = multitouch.apply(lambda row:
                                                        row['retention_touchpoints'] / row['touchpoints'] >= 0.50, axis=1)

    acquisition_converted = multitouch[multitouch['order_new_returning'] == 'new']
    retention_converted = multitouch[multitouch['order_new_returning'] == 'returning']

    retention_nonconv = multitouch[(multitouch['contains_conversion'] == 0) & (multitouch['majority_retention'])]

    acquisition_nonconv= multitouch[(multitouch['contains_conversion'] == 0) & ~(multitouch['majority_retention'])]

    acquisition = pd.concat([acquisition_converted, acquisition_nonconv])
    retention = pd.concat([retention_converted, retention_nonconv])

    return acquisition, retention


def output_table(data):
    model = MarkovAttribution(data=data, conv_col='contains_conversion')
    model.fit()
    model.all_removal_effects()

    re = pd.DataFrame(model.removal_effects.values(),
                      index=model.removal_effects.keys(),
                      columns=['Removal Effect'])

    re['Attributable Conversions'] = np.round(
        (re['Removal Effect'] / re['Removal Effect'].sum()) * model.data[model.conv_col].sum(), 1)

    re['Removal Effect'] = np.round(re['Removal Effect'], 3)
    re.sort_values(by='Removal Effect', ascending=False, inplace=True)

    return re


def plot_table(re, filename='table.png'):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')

    t = table(ax, re, loc='top')
    t.set_fontsize(12)
    t.scale(1.5, 1.5)

    plt.savefig(filename, bbox_inches='tight')


def main(a):
    # Get data tables
    acq, ret = download(FLAGS.start_date, FLAGS.end_date)

    acq_removal_effects = output_table(acq)
    plot_table(acq_removal_effects, filename='acquisition.png')

    ret_removal_effects = output_table(ret)
    plot_table(ret_removal_effects, filename='retention.png')

    # Build PDF
    report_name = FLAGS.report_name + '.pdf'
    canvas = Canvas(report_name, pagesize=letter)

    # Header
    canvas.setFont('Times-Bold', 18)
    canvas.setLineWidth(0.3)
    canvas.drawString(1 * inch, 10 * inch, 'Monthly Data-Driven Attribution Report')
    canvas.line(50,750,600,750)

    # Sub-header
    canvas.setFont('Times-Roman', 11)
    canvas.setLineWidth(0.05)
    canvas.drawString(1 * inch, 9.7 * inch, f'{FLAGS.start_date} to {FLAGS.end_date}')
    canvas.drawString(1 * inch, 9.3 * inch,'The following two tables describe 1. Removal Effect, the percentage of conversions which would ')
    canvas.drawString(1 * inch, 9.1 * inch, 'NOT have occurred without the channel and 2. Attributable Conversions, the number of conversions including assists ')
    canvas.drawString(1 * inch, 8.9 * inch, 'conversions including assists attributable to the channel in the past month.')


    # Acquisition
    canvas.setFont('Times-Bold', 12)
    canvas.drawString(1 * inch, 8.7 * inch, 'Acquisition')
    canvas.setFont('Times-Roman', 11)
    canvas.drawString(1 * inch, 8.7 * inch, '')
    canvas.drawImage('acquisition.png', 1 * inch, .1 * inch, width=350, height=600, anchor='c')
    canvas.showPage()

    # Retention
    canvas.setFont('Times-Bold', 12)
    canvas.drawString(1 * inch, 10 * inch, 'Retention')

    # Additional notes
    canvas.setFont('Times-Roman', 12)
    canvas.drawString(1 * inch, 9.75 * inch, '')
    canvas.drawImage('retention.png', 1 * inch, .2 * inch, width=360, height=600, anchor='c')

    canvas.save()


if __name__=='__main__':
    app.run(main)