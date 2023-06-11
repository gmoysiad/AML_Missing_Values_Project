# %%
import google_trends as gt
import matplotlib.pyplot as plt

from datetime import datetime

# %%
start = datetime(2014, 1, 1)
end = datetime(2015, 12, 31)

query='hilton hotel'
test = gt.Query(query=query, start=start, end=end, days=7)

# %%
test.timeline.plot()
plt.show()

# %%
rescaled_test = test.rescale(test.timeline, start, end, query)

# %%
print('end')