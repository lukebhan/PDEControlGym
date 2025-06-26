import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

class Simulation():

  def __init__(self):
    # dimensional parameters
    self.L = 200 #mm
    self.t = 600 #days
    self.dx = 1 #spatial size in mm
    self.dt = 1 #time step in days
    self.nt = int(round(self.t / self.dt)) + 1
    self.nx = int(round(self.L / self.dx)) + 1
    print(f"HathoutPaperSim has nt:{self.nt}, nx:{self.nx}")

    # experiment hyperparameters
    self.D = 0.2 #mm^2/day
    self.rho = 0.03 #cells/day
    self.alpha = 0.04
    self.alphaBetaRatio = 10 #constant at 10
    self.k = 1e5
    print(f"k:{self.k}")

    # state variables
    self.c = np.zeros((self.nt, self.nx)) # time x space grid
    self.cBenchmark = np.zeros((self.nt, self.nx)) # time x space grid
    self.cumulativeDose = np.zeros(self.nx)
    self.inRadiationDays = np.full(self.nt, False) # boolean[] for tracking therapy
    self.xScale = np.linspace(0, self.L, self.nx) # index -> space mapping
    self.tScale = np.linspace(0, self.t, self.nt) # index -> time mapping
    print(f"xScale: {self.xScale[:10]}. xScale length {len(self.xScale)}")
    print(f"tScale: {self.tScale[:10]}. tScale length {len(self.tScale)}")

    # simulation variables
    self.simulationDays = 0
    self.growthDays = 0
    self.therapyDays = 0
    self.postTherapyDays = 0
    self.T1DetectionRadius = None
    self.T2DetectionRadius = None
    self.firstTherapyDay = None
    self.firstPostTherapyDay = None
    self.cDeathDay = None
    self.cBenchmarkDeathDay = None

    # set initial condition
    c0 = 0.8 * self.k * np.exp(-0.25 * (self.xScale ** 2))
    self.c[0] = c0
    print(f"initial condition: {self.c[0][:10]}")
    self.cBenchmark[0] = c0

  def calcCNext(self, tIdx, dArray, inRadiation, target='c', DIT=None, DOT=None):
    print(f"Perform dimensionalized finite differencing for tIdx={tIdx}:")

    if (DIT is not None):
      print(f"\tDIT(inc. weekends) = {DIT}, DOT(exc. weekends) = {DOT}")
    
    # Calculate R term
    R = np.ones_like(dArray)
    if (inRadiation is True):
      self.cumulativeDose += dArray
      BED = (dArray + ((dArray ** 2) / (self.alphaBetaRatio)) )
      print(f"BED = {BED[0]}")
      S = np.exp(-self.alpha * BED)
      R = R - S
      print(f"S = e^(-{self.alpha} * {BED[0]}) = {S[0]} 	 R = {R[0]}   rho = {self.rho}")
      
    else:
      R = np.zeros_like(dArray)
      print(f"\tNOTHERAPY G = {R[0]:.3f}")
    
    # Finite differencing
    cCurr = getattr(self, target)[tIdx-1]
    cNext = np.zeros_like(cCurr)
    # With carrying capacity term
    cNext[1:self.nx-1] = ( cCurr[1:self.nx-1] + 
                          self.dt * (self.D * (((cCurr[2:self.nx] - 2 * cCurr[1:self.nx-1] + cCurr[0:self.nx-2]) / (self.dx ** 2)))
                                      + self.rho * cCurr[1:self.nx-1] * (np.ones_like(dArray)[1:self.nx-1] - (cCurr[1:self.nx-1] / self.k))
                                      - R[1:self.nx-1] * cCurr[1:self.nx-1] * (np.ones_like(dArray)[1:self.nx-1] - (cCurr[1:self.nx-1] / self.k))
                                    )
                          )

    cNext[0] = cNext[1]
    cNext[self.nx-1] = cNext[self.nx-2]
    cNext = np.clip(cNext, 0, self.k)

    getattr(self, target)[tIdx] = cNext
    return cNext
  
  # returns the greatest index visible on scan of detectionRatio
  def getTumorRadius(self, tIdx, detectionRatio, target='c'):
    densities = getattr(self, target)[tIdx]
    threshold = detectionRatio * self.k
    binaryMask = (densities >= threshold).astype(int)
    rightmostIdx = np.where(binaryMask == 1)[0].max() if np.any(binaryMask == 1) else None
    if rightmostIdx is not None:
      tumorRadius = self.xScale[rightmostIdx]
      return tumorRadius
    return None

  def runSimulation(self, detectionRadius=15):
    self.T1DetectionRadius = detectionRadius

    tIdx = 1

    # GROWTH STAGE START
    # Allow cancer to grow until T1Rad >= detectionRadius
    stage = "GROWTH"
    print(f"Begin pre-detection GROWTH stage")
    print(f"{'Stage':<15} {'tIdx':<5} {'T1TumorRadius':<15} {'T2TumorRadius':<15}")
    print("-" * 80)
    while (tIdx < self.nt):
      # set radiation dose d(xbar, tbar) = 0
      dArrayPlaceholder = np.zeros_like(self.c[0])
      self.calcCNext(tIdx, dArrayPlaceholder, False, target='c')

      # log metrics
      T1TumorRadius = self.getTumorRadius(tIdx, 0.8, target='c')
      T2TumorRadius = self.getTumorRadius(tIdx, 0.16, target='c')
      print(f"{stage:<15} {tIdx:<5} {f'{T1TumorRadius:.2f}' if T1TumorRadius is not None else 'None':<15} {f'{T2TumorRadius:.2f}' if T2TumorRadius is not None else 'None':<15}\n")

      # break if T1Rad >= detectionRadius
      if (T1TumorRadius is not None and T1TumorRadius >= detectionRadius):
        self.growthDays = tIdx
        self.T2DetectionRadius = T2TumorRadius
        print(f"GROWTH Break when tIdx={tIdx}. T1TumorRadius={T1TumorRadius:.2f} meets detectionRadius={detectionRadius}. T2TumorRadius={T2TumorRadius:.2f}")
        print(f"Summary: {self.growthDays} days spent in GROWTH stage\n")
        break

      tIdx += 1
    # GROWTH STAGE END
    
    # Initialize therapy schedule
    # 5 weekdays on, 2 weekend days off
    # phase 1: 28 treatment days: 1.8Gy to T2 region + 25mm
    # phase 2: 6 treatment days: 1.8Gy to T1 region + 20mm
    therapySchedule = [True, True, True, True, True, False, False]
    DIT = 0 #day counter (tracks weekends)
    DOT = 0 #treatment day counter
    totalTherapyDays = 34

    # THERAPY STAGE START
    # perform radiation therapy
    self.firstTherapyDay = tIdx+1
    print(f"Begin THERAPY stage on tIdx={tIdx+1} using tIdx={tIdx} for treatment reference")
    print(f"{'Stage':<15} {'Label':<5} {'tIdx':<5} {'T1TumorRadius':<15} {'T2TumorRadius':<15} {'treatmentRadius':<15}")
    print("-" * 80)
    while (DOT < totalTherapyDays and tIdx < self.nt):
      isRadiationDay = therapySchedule[DIT % 7]
      dArray = np.zeros_like(self.c[0])
      # use previous day metrics to compute radiation function d(xbar, tbar)
      T1TumorRadius = self.getTumorRadius(tIdx, 0.8, target='c')
      T2TumorRadius = self.getTumorRadius(tIdx, 0.16, target='c')
      # weekday -> perform radiation therapy
      if isRadiationDay:
        stage = "THERAPY"
        # determine therapy phase
        if DOT < 28:
          # T2 + 25mm radiation domain
          label = "T2"
          if (T2TumorRadius is None):
            treatmentRadius = -1
            print("Tumor is below detectable threshold. 0 treatment given")
          else:
            treatmentRadius = T2TumorRadius + 25
        else:
          # T1 + 20mm radiation domain
          label = "T1"
          if (T1TumorRadius is None):
            treatmentRadius = -1
            print("Tumor is below detectable threshold. 0 treatment given")
          else:
            treatmentRadius = T1TumorRadius + 20
        # construct dArray
        dArray[self.xScale <= treatmentRadius] = 1.8
        DOT += 1
      # weekend -> don't perform radiation therapy
      else:
        stage = "REST"
        label = None
        treatmentRadius = None

      # perform finite differencing
      tIdx += 1
      DIT += 1
      self.inRadiationDays[tIdx] = isRadiationDay
      self.calcCNext(tIdx, dArray, isRadiationDay, target='c', DIT=DIT, DOT=DOT)

      # log metrics
      T1TumorRadius = self.getTumorRadius(tIdx, 0.8, target='c')
      T2TumorRadius = self.getTumorRadius(tIdx, 0.16, target='c')
      print(f"{stage:<15} {f'{label}' if label is not None else 'None':<5} {tIdx:<5} {f'{T1TumorRadius:.2f}' if T1TumorRadius is not None else 'None':<15} {f'{T2TumorRadius:.2f}' if T2TumorRadius is not None else 'None':<15} {f'{treatmentRadius:.2f}' if treatmentRadius is not None else 'None':<15}\n")

    if (self.growthDays > 0):
      self.therapyDays = tIdx - self.growthDays
      print(f"Summary: {self.therapyDays} days spent in THERAPY stage\n")
    # THERAPY STAGE END

    # POST THERAPY STAGE START
    # Grow cancer for 200 more days, or until time domain exhausted
    tIdx += 1
    stage = "POST-THERAPY"
    print(f"Begin POST-THERAPY stage starting tIdx={tIdx}. Run simulation for 100 additional days or until time domain exhausted")
    self.firstPostTherapyDay = tIdx
    print(f"{'Stage':<15} {'tIdx':<5} {'T1TumorRadius':<15} {'T2TumorRadius':<15}")
    print("-" * 80)
    postTherapyEnd = min(tIdx + 200, self.nt)
    while (tIdx < postTherapyEnd):
      # set radiation dose d(xbar, tbar) = 0
      dArrayPlaceholder = np.zeros_like(self.c[0])
      self.calcCNext(tIdx, dArrayPlaceholder, False, target='c')

      # log metrics
      T1TumorRadius = self.getTumorRadius(tIdx, 0.8, target='c')
      T2TumorRadius = self.getTumorRadius(tIdx, 0.16, target='c')
      # set cDeathDay if T1TumorRadius >= 35mm
      if (self.cDeathDay is None and T1TumorRadius is not None and T1TumorRadius >= 35):
        self.cDeathDay = tIdx
      print(f"{stage:<15} {tIdx:<5} {f'{T1TumorRadius:.2f}' if T1TumorRadius is not None else 'None':<15} {f'{T2TumorRadius:.2f}' if T2TumorRadius is not None else 'None':<15}\n")
      tIdx += 1
    
    if (self.growthDays > 0 and self.therapyDays > 0):
      self.postTherapyDays = postTherapyEnd - self.firstPostTherapyDay
      print(f"Summary: {self.postTherapyDays} days spent in POST-THERAPY stage\n")
    # POST THERAPY STAGE END

    self.simulationDays = self.growthDays + self.therapyDays + self.postTherapyDays

    print(f"self.simulationDays {self.simulationDays}")
    print(f"self.growthDays {self.growthDays}")
    print(f"self.therapyDays {self.therapyDays}")
    print(f"self.postTherapyDays {self.postTherapyDays}")
    print(f"self.firstTherapyDay {self.firstTherapyDay}")
    print(f"self.firstPostTherapyDay {self.firstPostTherapyDay}")
    print(f"self.cDeathDay {self.cDeathDay}")


  # intended to be run after runSimulation as a benchmark (no treatment)
  def runSimulationBenchmark(self):
    tIdx = 1

    # Run only the growth stage
    stage = "GROWTH"
    print(f"{'Stage':<15} {'tIdx':<5} {'T1TumorRadius':<15} {'T2TumorRadius':<15}")
    print("-" * 80)
    while (tIdx <= self.simulationDays):
      # set radiation dose d(xbar, tbar) = 0
      dArrayPlaceholder = np.zeros_like(self.c[0])
      self.calcCNext(tIdx, dArrayPlaceholder, False, target='cBenchmark')

      # log metrics
      T1TumorRadius = self.getTumorRadius(tIdx, 0.8, target='cBenchmark')
      T2TumorRadius = self.getTumorRadius(tIdx, 0.16, target='cBenchmark')
      # set cBenchmarkDeathDay if T1TumorRadius >= 35mm
      if (self.cBenchmarkDeathDay is None and T1TumorRadius is not None and T1TumorRadius >= 35):
        self.cBenchmarkDeathDay = tIdx
      print(f"{stage:<15} {tIdx:<5} {f'{T1TumorRadius:.2f}' if T1TumorRadius is not None else 'None':<15} {f'{T2TumorRadius:.2f}' if T2TumorRadius is not None else 'None':<15}\n")
      tIdx += 1
    
    print(f"self.cBenchmarkDeathDay {self.cBenchmarkDeathDay}")

  # calculate difference between cBar and cBarBenchmark, calculate performance metrics  
  def runSimDiff(self):
    # print radius values and percent effect values
    print(f"{'tIdx':<5} {'T1SimRad':<10} {'T2SimRad':<15} {'T1BMRad':<10} {'T2BMRad':<15} {'PE:T1Sim':<10} {'PE:T2Sim':<15} {'PE:T1BM':<10} {'PE:T2BM':<15}")
    print("-" * 80)
    for tIdx in range(self.simulationDays+1):
      if (tIdx == self.firstTherapyDay):
        print("FIRST DAY OF THERAPY")
      if (tIdx == self.firstPostTherapyDay):
        print("THERAPY ENDS")

      T1TumorRadiusSim = self.getTumorRadius(tIdx, 0.8, target='c')
      T1TumorRadiusBM = self.getTumorRadius(tIdx, 0.8, target='cBenchmark')
      T2TumorRadiusSim = self.getTumorRadius(tIdx, 0.16, target='c')
      T2TumorRadiusBM = self.getTumorRadius(tIdx, 0.16, target='cBenchmark')

      PercentEffectT1Sim = (self.T1DetectionRadius - T1TumorRadiusSim) / self.T1DetectionRadius if T1TumorRadiusSim is not None else None
      PercentEffectT1BM = (self.T1DetectionRadius - T1TumorRadiusBM) / self.T1DetectionRadius if T1TumorRadiusBM is not None else None
      PercentEffectT2Sim = (self.T2DetectionRadius - T2TumorRadiusSim) / self.T2DetectionRadius if T2TumorRadiusSim is not None else None
      PercentEffectT2BM = (self.T2DetectionRadius - T2TumorRadiusBM) / self.T2DetectionRadius if T2TumorRadiusBM is not None else None

      print(f"{tIdx:<5} {f'{T1TumorRadiusSim:.2f}' if T1TumorRadiusSim is not None else 'None':<10} {f'{T2TumorRadiusSim:.2f}' if T2TumorRadiusSim is not None else 'None':<15} {f'{T1TumorRadiusBM:.2f}' if T1TumorRadiusBM is not None else 'None':<10} {f'{T2TumorRadiusBM:.2f}' if T2TumorRadiusBM is not None else 'None':<15} {f'{PercentEffectT1Sim:.2f}' if PercentEffectT1Sim is not None else 'None':<10} {f'{PercentEffectT2Sim:.2f}' if PercentEffectT2Sim is not None else 'None':<15} {f'{PercentEffectT1BM:.2f}' if PercentEffectT1BM is not None else 'None':<10} {f'{PercentEffectT2BM:.2f}' if PercentEffectT2BM is not None else 'None':<15}")

  def plotHeatmap(self, tIdx, target='c'):
    data = getattr(self, target)
    densities = data[tIdx]
    #scaling for lognorm
    densitiesLogNorm = densities + np.ones_like(densities)

    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = plt.get_cmap("inferno")
    norm = colors.LogNorm(vmin=1e0, vmax=1e5)
    im = ax.imshow([densitiesLogNorm], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmap, norm=norm)

    ax.set_yticks([])
    ax.set_xlabel("Radius From Center (mm)")
    ax.set_title(f"Tumor Density Heatmap (t={int( self.tScale[tIdx] / self.dt )} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})", fontsize=14, pad=10)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.5)
    cbar.set_label("Tumor Density (cells/mm³, log scale)", fontsize=12)
    cbar.ax.tick_params(which='both', length=0)

    plt.tight_layout()
    plt.show()

  # plots T1 scan
  # def plotT1(self, tIdx, target='c'):
  #   data = getattr(self, target)
  #   densities = data[tIdx]
  #   threshold = 0.8 * self.k
  #   binaryMask = (densities >= threshold).astype(int)

  #   fig, ax = plt.subplots(figsize=(10, 3))
  #   cmap = colors.ListedColormap(["white", "black"])
  #   norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)
  #   im = ax.imshow([binaryMask], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmap, norm=norm)

  #   ax.set_yticks([])
  #   ax.set_xlabel("Radius From Center (mm)")
  #   ax.set_title(f"T1 Region Classification (t={int( self.tScale[tIdx] / self.dt )} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})", fontsize=14, pad=10)

  #   cbar = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=[0.25, 0.75], pad=0.5)
  #   cbar.ax.set_xticklabels(['invisible', 'visible'])
  #   cbar.set_label("Tumor Density Threshold (≥ 0.8k)", fontsize=12)
  #   cbar.ax.tick_params(which='both', length=0)

  #   plt.tight_layout()
  #   plt.show()

  # plots T2 scan
  # def plotT2(self, tIdx, target='c'):
  #   data = getattr(self, target)
  #   densities = data[tIdx]
  #   threshold = 0.16 * self.k
  #   binaryMask = (densities >= threshold).astype(int)

  #   fig, ax = plt.subplots(figsize=(10, 3))
  #   cmap = colors.ListedColormap(["white", "black"])
  #   norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)
  #   im = ax.imshow([binaryMask], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmap, norm=norm)

  #   ax.set_yticks([])
  #   ax.set_xlabel("Radius From Center (mm)")
  #   ax.set_title(f"T2 Region Classification (t={int( self.tScale[tIdx] / self.dt )} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})", fontsize=14, pad=10)

  #   cbar = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=[0.25, 0.75], pad=0.5)
  #   cbar.ax.set_xticklabels(['invisible', 'visible'])
  #   cbar.set_label("Tumor Density Threshold (≥ 0.16k)", fontsize=12)
  #   cbar.ax.tick_params(which='both', length=0)

  #   plt.tight_layout()
  #   plt.show()

  def plotAllHeatmap(self, tIdx, target='c'):
    data = getattr(self, target)
    densities = data[tIdx]

    thresholds = {
      "T1": 0.8 * self.k,
      "T2": 0.16 * self.k
    }
    thresholdsStrings = {
      "T1": "≥ 0.8k",
      "T2": "≥ 0.16k"
    }
    masks = {
      "T1": (densities >= thresholds["T1"]).astype(int),
      "T2": (densities >= thresholds["T2"]).astype(int)
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.5})
    titles = [
      f"Tumor Density Heatmap (t={int( self.tScale[tIdx] / self.dt )} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})",
      f"T1 Region Classification (t={int( self.tScale[tIdx] / self.dt )} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})",
      f"T2 Region Classification (t={int( self.tScale[tIdx] / self.dt )} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})"
    ]

    # Heatmap
    cmapHeatmap = plt.get_cmap("inferno")
    normHeatmap = colors.LogNorm(vmin=1e0, vmax=1e5)
    #scaling for lognorm
    densitiesLogNorm = densities + np.ones_like(densities)
    im0 = axes[0].imshow([densitiesLogNorm], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmapHeatmap, norm=normHeatmap)
    axes[0].set_yticks([])    
    axes[0].set_xlabel("Radius From Center (mm)")
    axes[0].set_title(titles[0], fontsize=12)
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.25)
    cbar0.set_label("Tumor Density (cells/mm³, log scale)")
    cbar0.ax.tick_params(which='both', length=0)

    # T1 + T2
    cmapBinary = colors.ListedColormap(["white", "black"])
    normBinary = colors.BoundaryNorm([0, 0.5, 1], cmapBinary.N)
    for i, label in enumerate(["T1", "T2"]):
      im = axes[i+1].imshow([masks[label]], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmapBinary, norm=normBinary)
      axes[i+1].set_yticks([])
      axes[i+1].set_xlabel("Radius From Center (mm)")
      axes[i+1].set_title(titles[i+1], fontsize=12)
      cbar = fig.colorbar(im, ax=axes[i+1], orientation="horizontal", ticks=[0.25, 0.75], pad=0.25)
      cbar.ax.set_xticklabels(['invisible', 'visible'])
      cbar.set_label(f"Tumor Density Threshold ({thresholdsStrings[label]})")
      cbar.ax.tick_params(which='both', length=0)
    
    plt.show()

  # plots line graph of T1 radius across all time indices
  def plotT1Line(self, target='c'):
    if (self.simulationDays == 0):
      raise ValueError('Run simulation first before plotting')

    data = getattr(self, target)
    detectionRatio = 0.8

    timeAxis = np.linspace(0, self.simulationDays, self.simulationDays + 1)
    radiiOverTime = np.zeros(self.simulationDays + 1)
    for tIdx in range(self.simulationDays + 1):
      radius = self.getTumorRadius(tIdx, detectionRatio, target=target)
      radiiOverTime[tIdx] = radius if (radius is not None) else 0

    fig, ax = plt.subplots()

    if (target == 'c'):
      b0 = 0
      b1 = self.firstTherapyDay
      b2 = self.firstPostTherapyDay
      b3 = self.simulationDays + 1
      for start, end, color in [
        (b0, b1, 'grey'),
        (b1, b2, 'red'),
        (b2, b3, 'black'),
      ]:
        if end > start:
          ax.plot(timeAxis[start:end], radiiOverTime[start:end], color=color, label='label')
    else:
      ax.plot(timeAxis, radiiOverTime, color='black')

    ax.set_title('T1 Visible Radius vs. Time')
    ax.set_xlabel(f'Time (1 tIdx = 1 day)')
    ax.set_ylabel('T1 Visible Radius (mm)')
    ax.grid(True)
    plt.xlim(0, self.simulationDays + 1)
    plt.show()

  # plots line graph of T2 radius across all time indices
  def plotT2Line(self, target='c'):
    if (self.simulationDays == 0):
      raise ValueError('Run simulation first before plotting')

    data = getattr(self, target)
    detectionRatio = 0.16

    timeAxis = np.linspace(0, self.simulationDays, self.simulationDays + 1)
    radiiOverTime = np.zeros(self.simulationDays + 1)
    for tIdx in range(self.simulationDays + 1):
      radius = self.getTumorRadius(tIdx, detectionRatio, target=target)
      radiiOverTime[tIdx] = radius if (radius is not None) else 0

    fig, ax = plt.subplots()

    if (target == 'c'):
      b0 = 0
      b1 = self.firstTherapyDay
      b2 = self.firstPostTherapyDay
      b3 = self.simulationDays + 1
      for start, end, color in [
        (b0, b1, 'grey'),
        (b1, b2, 'red'),
        (b2, b3, 'black'),
      ]:
        if end > start:
          ax.plot(timeAxis[start:end], radiiOverTime[start:end], color=color, label='label')
    else:
      ax.plot(timeAxis, radiiOverTime, color='black')

    ax.set_title('T2 Visible Radius vs. Time')
    ax.set_xlabel(f'Time (1 tIdx = 1 day)')
    ax.set_ylabel('T2 Visible Radius (mm)')
    ax.grid(True)
    plt.xlim(0, self.simulationDays + 1)
    plt.show()

  def plotAllLines(self):
    if (self.simulationDays == 0):
      raise ValueError('Run simulation first before plotting')
    
    timeAxis = np.linspace(0, self.simulationDays, self.simulationDays + 1)
    radiiT1C = np.zeros(self.simulationDays + 1)
    radiiT2C = np.zeros(self.simulationDays + 1)
    radiiT1CBenchmark = np.zeros(self.simulationDays + 1)
    radiiT2CBenchmark = np.zeros(self.simulationDays + 1)

    for tIdx in range(self.simulationDays + 1):
      rT1C = self.getTumorRadius(tIdx, 0.8, target='c')
      radiiT1C[tIdx] = rT1C if (rT1C is not None) else 0
      rT2C = self.getTumorRadius(tIdx, 0.16, target='c')
      radiiT2C[tIdx] = rT2C if (rT2C is not None) else 0
      rT1CBenchmark = self.getTumorRadius(tIdx, 0.8, target='cBenchmark')
      radiiT1CBenchmark[tIdx] = rT1CBenchmark if (rT1CBenchmark is not None) else 0
      rT2CBenchmark = self.getTumorRadius(tIdx, 0.16, target='cBenchmark')
      radiiT2CBenchmark[tIdx] = rT2CBenchmark if (rT2CBenchmark is not None) else 0

    fig, ax = plt.subplots()

    # Plot lethal threshold line
    ax.plot(timeAxis, 35*np.ones_like(radiiT1C), color='black', label=f'T1 lethal threshold')

    # Plot T1C and T2C
    ax.plot(timeAxis, radiiT1C, color='darkred', label=f'T1 Radius C')
    ax.plot(timeAxis, radiiT2C, color='red', label=f'T2 Radius C')

    # Plot T1CBenchmark and T2CBenchmark
    ax.plot(timeAxis, radiiT1CBenchmark, color='rosybrown', label=f'T1 Radius CBenchmark')
    ax.plot(timeAxis, radiiT2CBenchmark, color='mistyrose', label=f'T2 Radius CBenchmark')

    ax.set_title('T1 and T2 Visible Radius vs. Time')
    ax.set_xlabel(f'Time (1 tIdx = 1 day)')
    ax.set_ylabel('Visible Radius (mm)')
    ax.grid(True)
    plt.xlim(0, self.simulationDays + 1)
    ax.legend()
    plt.show()


  def plotDensity(self, *tIdxs, target='c'):
    data = getattr(self, target)
    fig, ax = plt.subplots()

    colors = plt.cm.tab10.colors

    for idx, tIdx in enumerate(tIdxs):
      densities = data[tIdx]
      label = f't = {int(self.tScale[tIdx] / self.dt)} days'
      ax.plot(self.xScale, densities, color=colors[idx], label=label)

    ax.set_title(f'Density Plot Over Time')
    ax.set_xlabel(f'X (1 unit = {self.dx} mm)')
    ax.set_ylabel('Cell density (cells / mm^3)')
    ax.grid(True)
    plt.xlim(0, self.nx + 1)
    plt.ylim(0, self.k)
    ax.legend()
    plt.show()

  def plot3D(self, target='c'):
    data = getattr(self, target)
    fig = plt.figure(figsize=(8, 3))
    fig.suptitle(f"Evolution of Tumor Density Over Time target={target}", fontsize=12)
    ax = fig.add_subplot(projection='3d')

    spatial = np.linspace(0, self.L, self.nx)
    temporal = np.linspace(0, self.simulationDays, self.simulationDays+1)
    dataCut = data[:self.simulationDays+1,:]
    
    meshx, mesht = np.meshgrid(spatial, temporal)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
      axis._axinfo['axisline']['linewidth'] = 1
      axis._axinfo['axisline']['color'] = "b"
      axis._axinfo['grid']['linewidth'] = 0.2
      axis._axinfo['grid']['linestyle'] = "--"
      axis._axinfo['grid']['color'] = "#d1d1d1"
      axis.set_pane_color((1,1,1))

    ax.set_xlabel("x", labelpad=5)
    ax.set_ylabel("Time", labelpad=5)
    ax.set_zlabel("Cell density (cells / mm^3", labelpad=5)
    ax.tick_params(axis='x', which='major', labelsize=8, pad=-2)
    ax.tick_params(axis='y', which='major', labelsize=8, pad=-2)
    ax.tick_params(axis='z', which='major', labelsize=8, pad=2)
    ax.set_xlim(0, self.L)
    ax.set_ylim(0, self.simulationDays)

    ax.view_init(elev=30, azim=-60)

    # plot surface
    surf = ax.plot_surface(meshx, mesht, dataCut, edgecolor="black", lw=0.2, alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
    
    # plot initial condition
    t0 = np.zeros_like(spatial) # array of time coords
    vals = dataCut[0]
    ax.plot(spatial, t0, vals, color="red", lw=0.1, antialiased=False, rasterized=False, zorder=10)

    # plot death day
    death_day = getattr(self, f"{target}DeathDay")
    if (death_day is not None):
      t_death = np.full_like(spatial, death_day) # array of time coords
      vals = dataCut[death_day]
      ax.plot(spatial, t_death, vals, color="red", lw=0.1, antialiased=False, rasterized=False, zorder=10)

    

  # plots cumulative dose with respect to radial distance
  def plotCumulativeDose(self):
    if (self.simulationDays == 0):
      raise ValueError('Run simulation first before plotting')

    data = self.cumulativeDose
    radialDistance = self.xScale

    fig, ax = plt.subplots()
    ax.plot(radialDistance, data, color='black')

    ax.set_title('Cumulative Dose vs. Radial Distance')
    ax.set_xlabel(f'Radial Distance (1 unit = {self.dx:.2f} mm)')
    ax.set_ylabel('Cumulative Dose (Gy)')
    ax.grid(True)
    plt.xlim(0, self.L + 1)
    plt.show()

  '''
  Usecase: 
  ani = ndSim.animateAllHeatmap()

  Display in Jupyter notebook:
  HTML(ani.to_jshtml())

  Save as Mp4:
  ani.save("animations/all_heatmap.mp4", writer="ffmpeg", fps=30)
  '''
  # def animateHeatmap(self, msPerFrame=50, target='c'):
  #   data = getattr(self, target)
  #   initialDensities = data[0]

  #   fig, ax = plt.subplots(figsize=(10, 3))
  #   fig.subplots_adjust(bottom=0.3)
  #   cmap = plt.get_cmap("inferno")
  #   norm = colors.LogNorm(vmin=1e0, vmax=1e5)
  #   im = ax.imshow([initialDensities], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmap, norm=norm)

  #   ax.set_yticks([])
  #   ax.set_xlabel("Radius From Center (mm)")
  #   title = ax.set_title(f"Tumor Density Heatmap (t=0 days, therapy={target == 'c' and self.inRadiationDays[0]})", fontsize=14, pad=10)

  #   cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.5)
  #   cbar.set_label("Tumor Density (cells/mm³, log scale)", fontsize=12)
  #   cbar.ax.tick_params(which='both', length=0)

  #   def update(tIdx):
  #     densities = data[tIdx]
  #     im.set_data([densities])
  #     title.set_text(f"Tumor Density Heatmap (t={int(self.tScale[tIdx] / self.dt)} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})")
  #     return [im, title]

  #   ani = animation.FuncAnimation(fig, update, frames=self.simulationDays, blit=False, interval=msPerFrame)
  #   return ani
  
  # def animateT1(self, msPerFrame=50, target='c'):
  #   data = getattr(self, target)
  #   initialDensities = data[0]
  #   initialBinaryMask = (initialDensities >= threshold).astype(int)

  #   fig, ax = plt.subplots(figsize=(10, 3))
  #   fig.subplots_adjust(bottom=0.3)
  #   cmap = colors.ListedColormap(["white", "black"])
  #   norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)
  #   threshold = 0.8 * self.k
  #   im = ax.imshow([initialBinaryMask], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmap, norm=norm)

  #   ax.set_yticks([])
  #   ax.set_xlabel("Radius From Center (mm)")
  #   title = ax.set_title(f"T1 Region Classification (t=0 days, therapy={target == 'c' and self.inRadiationDays[0]})", fontsize=14, pad=10)

  #   cbar = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=[0.25, 0.75], pad=0.5)
  #   cbar.ax.set_xticklabels(['invisible', 'visible'])
  #   cbar.set_label("Tumor Density Threshold (≥ 0.8k)", fontsize=12)
  #   cbar.ax.tick_params(which='both', length=0)

  #   def update(tIdx):
  #     densities = data[tIdx]
  #     binaryMask = (densities >= threshold).astype(int)
  #     im.set_data([binaryMask])
  #     title.set_text(f"T1 Region Classification (t={int(self.tScale[tIdx] / self.dt)} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})")
  #     return [im, title]

  #   ani = animation.FuncAnimation(fig, update, frames=self.simulationDays, blit=False, interval=msPerFrame)
  #   return ani
  
  # def animateT2(self, msPerFrame=50, target='c'):
  #   data = getattr(self, target)
  #   initialDensities = data[0]
  #   initialBinaryMask = (initialDensities >= threshold).astype(int)

  #   fig, ax = plt.subplots(figsize=(10, 3))
  #   fig.subplots_adjust(bottom=0.3)
  #   cmap = colors.ListedColormap(["white", "black"])
  #   norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)
  #   threshold = 0.16 * self.k
  #   im = ax.imshow([initialBinaryMask], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmap, norm=norm)

  #   ax.set_yticks([])
  #   ax.set_xlabel("Radius From Center (mm)")
  #   title = ax.set_title(f"T2 Region Classification (t=0 days, therapy={target == 'c' and self.inRadiationDays[0]})", fontsize=14, pad=10)

  #   cbar = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=[0.25, 0.75], pad=0.5)
  #   cbar.ax.set_xticklabels(['invisible', 'visible'])
  #   cbar.set_label("Tumor Density Threshold (≥ 0.16k)", fontsize=12)
  #   cbar.ax.tick_params(which='both', length=0)

  #   def update(tIdx):
  #     densities = data[tIdx]
  #     binaryMask = (densities >= threshold).astype(int)
  #     im.set_data([binaryMask])
  #     title.set_text(f"T2 Region Classification (t={int(self.tScale[tIdx] / self.dt)} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})")
  #     return [im, title]

  #   ani = animation.FuncAnimation(fig, update, frames=self.simulationDays, blit=False, interval=msPerFrame)
  #   return ani

  def animateAllHeatmap(self, msPerFrame=50, target='c'):
    data = getattr(self, target)
    initialDensities = data[0]
    T1Threshold = 0.8 * self.k
    T2Threshold = 0.16 * self.k
    initialT1Mask = (initialDensities >= T1Threshold).astype(int)
    initialT2Mask = (initialDensities >= T2Threshold).astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.5})
    fig.subplots_adjust(bottom=0.1)

    # Tumor Density Heatmap
    cmapHeatmap = plt.get_cmap("inferno")
    normHeatmap = colors.LogNorm(vmin=1e0, vmax=1e5)
    im0 = axes[0].imshow([initialDensities], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmapHeatmap, norm=normHeatmap)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Radius From Center (mm)")
    title0 = axes[0].set_title(f"Tumor Density Heatmap (t=0 days, therapy={target == 'c' and self.inRadiationDays[0]})", fontsize=14, pad=10)
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.5)
    cbar0.set_label("Tumor Density (cells/mm³, log scale)", fontsize=12)
    cbar0.ax.tick_params(which='both', length=0)

    # T1 Mask
    cmapBinary = colors.ListedColormap(["white", "black"])
    normBinary = colors.BoundaryNorm([0, 0.5, 1], cmapBinary.N)
    im1 = axes[1].imshow([initialT1Mask], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmapBinary, norm=normBinary)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Radius From Center (mm)")
    title1 = axes[1].set_title(f"T1 Region Classification (t=0 days, therapy={target == 'c' and self.inRadiationDays[0]})", fontsize=14, pad=10)
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", ticks=[0.25, 0.75], pad=0.5)
    cbar1.ax.set_xticklabels(['invisible', 'visible'])
    cbar1.set_label("Tumor Density Threshold (≥ 0.8k)", fontsize=12)
    cbar1.ax.tick_params(which='both', length=0)

    # T2 Mask
    im2 = axes[2].imshow([initialT2Mask], aspect="auto", extent=[self.xScale[0], self.xScale[-1], 0, 1], cmap=cmapBinary, norm=normBinary)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Radius From Center (mm)")
    title2 = axes[2].set_title(f"T2 Region Classification (t=0 days, therapy={target == 'c' and self.inRadiationDays[0]})", fontsize=14, pad=10)
    cbar2 = fig.colorbar(im2, ax=axes[2], orientation="horizontal", ticks=[0.25, 0.75], pad=0.5)
    cbar2.ax.set_xticklabels(['invisible', 'visible'])
    cbar2.set_label("Tumor Density Threshold (≥ 0.16k)", fontsize=12)
    cbar2.ax.tick_params(which='both', length=0)

    def update(tIdx):
      densities = data[tIdx]
      im0.set_data([densities])
      title0.set_text(f"Tumor Density Heatmap (t={int(self.tScale[tIdx] / self.dt)} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})")
    
      T1Mask = (densities >= T1Threshold).astype(int)
      im1.set_data([T1Mask])
      title1.set_text(f"T1 Region Classification (t={int(self.tScale[tIdx] / self.dt)} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})")

      T2Mask = (densities >= T2Threshold).astype(int)
      im2.set_data([T2Mask])
      title2.set_text(f"T2 Region Classification (t={int(self.tScale[tIdx] / self.dt)} days, therapy={target == 'c' and self.inRadiationDays[tIdx]})")

      return [im0, title0, im1, title1, im2, title2]
    
    ani = animation.FuncAnimation(fig, update, frames=self.simulationDays, blit=False, interval=msPerFrame)
    return ani
