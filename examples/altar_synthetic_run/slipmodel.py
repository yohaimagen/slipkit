#!/usr/bin/env python3
import altar
import altar.models.seismic

class SlipModel(altar.shells.application,
                family='altar.applications.slipmodel'):
    model = altar.models.model(default='altar.models.seismic.static')

app = SlipModel(name='slipmodel')
status = app.run()
raise SystemExit(status)
