# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# # Example data points
# methods_10 = ['Ours\n52.18', 'D-Ours\n44.17', 'NI\n76.04', 'DI\n197.33', 'PII\n220.59']  # X-axis values
# fid_10 = [52.18, 44.17, 76.04,197.33,220.59]
# iteration_10 = [2060, 10300,400000,400000,320000]


# methods_100 = ['Ours\n31.39', 'D-Ours\n28.15', 'NI\n62.90', 'DI\n151.52', 'PII\n229.72']  # X-axis values
# fid_100 = [31.39, 28.15, 62.9,151.52,229.72]
# iteration_100 = [20600, 103000,800000,800000,320000]

# # Create a figure and a grid of subplots
# fig = plt.figure(figsize=(12, 6))
# gs = GridSpec(3, 2, height_ratios=[1, 0.1,0.1, 1])  # The second row is the 'cut' in the fid_10-axis

# # Upper subplots
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])

# # Lower subplots
# ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
# ax4 = fig.add_subplot(gs[3, 1], sharex=ax2)

# # Plot upper parts
# # for ax in [ax1, ax2]:
# ax1.scatter(iteration_10[3:], fid_10[3:], color='grey', s=50, marker='o', zorder=5)
# for i, label in enumerate(methods_10[3:]):
#     ax1.text(iteration_10[i+2], fid_10[i+2], label, fontsize=9, ha='left', va='top')
# ax1.set_ylim(150, 240)
# ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # hide the iteration_10-tick methods_10 on top


# ax2.scatter(iteration_100[3:], fid_100[3:], color='grey', s=50, marker='o', zorder=5)
# for i, label in enumerate(methods_100[3:]):
#     ax2.text(iteration_100[i+3], fid_100[i+3], label, fontsize=9, ha='left', va='top')
# ax2.set_ylim(150, 240)
# ax2.xaxis.tick_top()
# ax2.tick_params(labeltop=False)  # hide the iteration_10-tick methods_10 on top

# # Plot lower parts
# # for ax in [ax3, ax4]:
# ax3.scatter(iteration_10[:2], fid_10[:2], color='magenta', s=100, marker='*', zorder=5)
# ax3.scatter(iteration_10[2], fid_10[2], color='grey', s=50, marker='o', zorder=5)
# for i, label in enumerate(methods_100[:2]):
#     ax3.text(iteration_10[i], fid_10[i], label, fontsize=9, ha='left', va='top',weight='bold')
# ax3.text(iteration_10[2], fid_10[2], methods_10[2], fontsize=9, ha='left', va='top')
# ax3.set_ylim(25, 80)

# ax4.scatter(iteration_100[:2], fid_100[:2], color='magenta', s=100, marker='*', zorder=5)
# ax4.scatter(iteration_100[2], fid_100[2], color='grey', s=50, marker='o', zorder=5)
# for i, label in enumerate(methods_100[:2]):
#     ax4.text(iteration_100[i], fid_100[i], label, fontsize=9, ha='left', va='top',weight='bold')
# ax4.text(iteration_100[2], fid_100[2], methods_100[2], fontsize=9, ha='left', va='top')
# ax4.set_ylim(25, 80)

# # Set scale for iteration_10-axis
# for ax in [ax3, ax4]:  # Only need to set for the lower axes
#     ax.set_xscale('linear')

# # Axes methods_10 for lower plots
# ax3.set_xlabel('Iteration',fontsize=15)
# ax4.set_xlabel('Iteration',fontsize=15)
# ax1.set_title('CIFAR10',fontsize=20)
# ax2.set_title('CIFAR100',fontsize=20)
# ax3.set_ylabel('FID',fontsize=15)
# ax4.set_ylabel('FID',fontsize=15)

# # Hide the spines between ax1 and ax3, ax2 and ax4
# for ax in [ax1, ax2]:
#     ax.spines['bottom'].set_visible(False)

# for ax in [ax3, ax4]:
#     ax.spines['top'].set_visible(False)

# # Show the plot with a tight layout
# plt.tight_layout()
# plt.savefig(f'inversion_cifar10_cifar100_computational_overhead.png')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.ticker as ticker

# Example data points
methods_10 = ['Ours\n(52.18)', 'D-Ours\n(44.17)', 'NI\n(76.04)', 'DI\n(197.33)', 'PII\n(220.59)']
fid_10 = [52.18, 44.17, 76.04, 197.33, 220.59]
iteration_10 = [2060, 10300, 400000, 400000, 560000]

methods_100 = ['Ours\n(31.39)', 'D-Ours\n(28.15)', 'NI\n(62.90)', 'DI\n(151.52)', 'PII\n(229.72)']
fid_100 = [31.39, 28.15, 62.9, 151.52, 229.72]
iteration_100 = [20600, 103000, 800000, 800000, 560000]

# Create a figure and a grid of subplots
# fig = plt.figure(figsize=(12, 5))
# gs = GridSpec(4, 2, height_ratios=[1, 0.1, 0.1, 1])  # Adjusted the 'cut' in the fid-axis
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,5))


# Function to format the x-axis
def format_xaxis(ax, x_scale, labels_scale):
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_scale))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/labels_scale)}k'))



# Upper subplots





# Lower subplots
# ax3 = fig.add_subplot(gs[3, 0])
# ax4 = fig.add_subplot(gs[3, 1])

# ax1 = fig.add_subplot(gs[0, 0], sharex=ax3)
# ax2 = fig.add_subplot(gs[0, 1], sharex=ax4)

# ax5 = fig.add_subplot(gs[2, 0])
# ax6 = fig.add_subplot(gs[2, 1])



# Format the x-axis for CIFAR10
# format_xaxis(ax3, 100000, 100000)

# # Format the x-axis for CIFAR100
# format_xaxis(ax4, 200000, 100000)

# Hide x-axis labels for the upper axes and show for the lower axes
# ax1.tick_params(labelbottom=False)
# ax2.tick_params(labelbottom=False)
# ax3.tick_params(labelbottom=True)
# ax4.tick_params(labelbottom=True)


# Plot upper parts
ax1.scatter(iteration_10[3:], fid_10[3:], color='grey', s=200, marker='o', zorder=5)
ax2.scatter(iteration_100[3:], fid_100[3:], color='grey', s=200, marker='o', zorder=5)

# Plot lower parts
ax1.scatter(iteration_10[:2], fid_10[:2], color=['r', 'r'], s=[300, 300],edgecolors='black',  marker='*', zorder=5)
# ax3.plot(iteration_10[:2], fid_10[:2], color='orange',lw=1,zorder=5)
ax1.scatter(iteration_10[2], fid_10[2], color=['grey'], s=200, marker='o', zorder=5)
ax2.scatter(iteration_100[:2], fid_100[:2], color=['r', 'r'], s=[300, 300],edgecolors='black',  marker='*', zorder=5)
ax2.scatter(iteration_100[2], fid_100[2], color=['grey'], s=200, marker='o', zorder=5)
# ax4.plot(iteration_100[:2], fid_100[:2], color='orange',lw=1,zorder=5)

for i, label in enumerate(methods_10[3:]):
    # ax1.text(iteration_10[i+3], fid_10[i+3], label, fontsize=9, ha='left')
    ax1.annotate(label,
            xy=(iteration_10[i+3], fid_10[i+3]),  # theta, radius
            xytext=(-40, -25),    # fraction, fraction
            textcoords='offset points',
            xycoords='data',
             fontsize=12)
# for i, label in enumerate(methods_100[3]):
    # ax2.text(iteration_100[i+3], fid_100[i+3], label, fontsize=9, ha='left')
ax2.annotate(methods_100[3],
        xy=(iteration_100[3], fid_100[3]),  # theta, radius
        xytext=(-40, -25),    # fraction, fraction
            textcoords='offset points',
            xycoords='data',
             fontsize=12)
# for i, label in enumerate(methods_100[4]):
    # ax2.text(iteration_100[i+3], fid_100[i+3], label, fontsize=9, ha='left')
ax2.annotate(methods_100[4],
        xy=(iteration_100[4], fid_100[4]),  # theta, radius
       xytext=(-40, -25),    # fraction, fraction
            textcoords='offset points',
            xycoords='data',
             fontsize=12)

# for i, label in enumerate(methods_10[:2]):
    # ax3.text(iteration_10[i], fid_10[i], label, fontsize=9, ha='left',weight='bold')
ax1.annotate(methods_10[0],
        xy=(iteration_10[0], fid_10[0]),  # theta, radius
        xytext=(40, 30),    # fraction, fraction
        textcoords='offset points',
        xycoords='data',
        color='red',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3', 
                            color='black'),
        weight='bold',
        fontsize=18
        )
ax1.annotate(methods_10[1],
        xy=(iteration_10[1], fid_10[1]),  # theta, radius
        xytext=(60, -10),    # fraction, fraction
        textcoords='offset points',
        xycoords='data',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3', 
                            color='black'),
        color='red',
        weight='bold',
        fontsize=18
        )
# ax3.text(iteration_10[2], fid_10[2], methods_10[2], fontsize=9, ha='left')

ax1.annotate(methods_10[2],
        xy=(iteration_10[2], fid_10[2]),  # theta, radius
        xytext=(-40, -25),    # fraction, fraction
            textcoords='offset points',
            xycoords='data',
             fontsize=12)
# for i, label in enumerate(methods_100[:2]):
    # ax4.text(iteration_100[i], fid_100[i], label, fontsize=9, ha='left',weight='bold')
ax2.annotate(methods_100[0],
        xy=(iteration_100[0], fid_100[0]),  # theta, radius
        xytext=(20, 50),    # fraction, fraction
        textcoords='offset points',
        xycoords='data',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3', 
                            color='black'),
        weight='bold',
        color='red',
        fontsize=18
        )

ax2.annotate(methods_100[1],
        xy=(iteration_100[1], fid_100[1]),  # theta, radius
        xytext=(60, 15),    # fraction, fraction
        textcoords='offset points',
        xycoords='data',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3', 
                            color='black'),
        weight='bold',
        color='red',
        fontsize=18
        )
# ax4.text(iteration_100[2], fid_100[2], methods_100[2], fontsize=9, ha='left')
ax2.annotate(methods_100[2],
        xy=(iteration_100[2], fid_100[2]),  # theta, radius
        xytext=(-40, -25),    # fraction, fraction
            textcoords='offset points',
            xycoords='data',
             fontsize=12)
# Set y-limits for the upper and lower plots
# ax1.set_ylim(fid_100[-2]-10, max(fid_100) + 10)
# ax2.set_ylim(fid_100[-2]-10, max(fid_100) + 10)
# ax3.set_ylim(min(fid_100) - 10, 90)
# ax4.set_ylim(min(fid_100) - 10, 90)

# # Set x-limits for the lower plots (this will also affect the upper plots due to shared x-axis)
# ax3.set_xlim(0, max(iteration_10)+ 10000)
# ax4.set_xlim(0, max(iteration_100)+20000)

# # Set custom x-ticks and labels for CIFAR10
# ax3.set_xticks(np.arange(0, max(iteration_10), 2000))
ax1.set_xticklabels([f'{i//1000}k' for i in np.arange(-100000, 600001, 100000)])

# # Set custom x-ticks and labels for CIFAR100
# ax4.set_xticks(np.arange(0, max(iteration_100), 20000))
ax2.set_xticklabels([f'{i//1000}k' for i in np.arange(-100000,800001, 100000)])

# # Axes methods_10 for lower plots
ax1.set_xlabel('Iteration',fontsize=15)
ax2.set_xlabel('Iteration',fontsize=15)
ax1.set_title('CIFAR10',fontsize=20)
ax2.set_title('CIFAR100',fontsize=20)
ax1.set_ylabel('FID',fontsize=15)
ax2.set_ylabel('FID',fontsize=15)

# fid_label_x = 0.085  # x location in figure coordinates
# fid_label_y = 0.48  # y location in figure coordinates
# plt.figtext(fid_label_x, fid_label_y, 'FID', ha='center', va='center', fontsize=15)

# fid_label_x = 0.5  # x location in figure coordinates
# fid_label_y = 0.45  # y location in figure coordinates
# plt.figtext(fid_label_x, fid_label_y, 'FID', ha='center', va='center', fontsize=12)

# Hide the spines between ax1 and ax3, ax2 and ax4
# ax1.xaxis.set_major_locator(ticker.NullLocator())
# ax2.xaxis.set_major_locator(ticker.NullLocator())
# xax = ax1.get_xaxis()
# xax = xax.set_visible(False)
# xax = ax2.get_xaxis()
# xax = xax.set_visible(False)
# for ax in [ax1, ax2]:
#     ax.spines['bottom'].set_visible(False)
#     # ax.spines['top'].set_visible(False)

# for ax in [ax3, ax4]:
#     ax.spines['top'].set_visible(False)

# Show the plot with a tight layout
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.001)  # Adjust the space to make the gap smaller

ax1.grid()
ax2.grid()


# Save the plot
plt.savefig('inversion_cifar10_cifar100_computational_overhead.pdf',dpi=300)

# Show the plot
plt.show()
