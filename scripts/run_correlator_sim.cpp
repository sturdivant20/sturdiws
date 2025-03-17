
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <navtools/constants.hpp>
#include <navtools/frames.hpp>
#include <random>
#include <satutils/atmosphere.hpp>
#include <satutils/gnss-constants.hpp>
#include <sturdins/kns.hpp>
#include <vector>

#include "navtools/attitude.hpp"
#include "sim_common.hpp"

void vt_sim(
    SimParam &conf, spdlog::stopwatch &sw, std::shared_ptr<spdlog::logger> logger, int seed) {
  // constants
  double lambda = navtools::LIGHT_SPEED<double> / satutils::GPS_L1_FREQUENCY<double>;
  double beta = navtools::LIGHT_SPEED<double> / satutils::GPS_CA_CODE_RATE<double>;
  double kappa = satutils::GPS_CA_CODE_RATE<double> / satutils::GPS_L1_FREQUENCY<double>;
  double R2Dsq = navtools::RAD2DEG<> * navtools::RAD2DEG<>;

  // 2) Parse satellite ephemeris
  std::vector<satutils::KeplerElements<double>> elem;
  std::vector<satutils::KlobucharElements<double>> klob;
  ParseEphem(conf, elem, klob);

  // 3) Skip to desired time in truth data
  Truth truth_k, truth_kp1;
  std::ifstream fid = OpenDataFile(conf, truth_k);
  Eigen::Vector3d rpy_est, xyz, xyzv, lla, nedv;

  // 4) Create desired random noise generator
  std::mt19937_64 noise_gen;
  noise_gen.seed(seed);
  std::normal_distribution<double> noise_dist(0.0, 1.0);

  // 5) Initialize simulators
  Simulators sim = CreateSimulators(conf, elem, klob, lambda, noise_gen, noise_dist);
  sim.clk.GetCurrentState(truth_k.cb, truth_k.cd);

  // 6) initialize navigator/estimators
  Eigen::VectorXd true_cno(conf.n_sv), est_cno(conf.n_sv),
      j2s{std::pow(10.0, conf.j2s / 10.0) * Eigen::VectorXd::Ones(conf.n_sv)};
  CnoEstimator cno_estimator(conf.n_sv);
  sturdins::Kns nav = InitNavigator(conf, truth_k, noise_gen, noise_dist);

  // 7) Initialize tracking loop states
  TrackingStates true_state_k = InitTrackingState(
      conf,
      sim.obs,
      conf.init_tow,
      truth_k.lat,
      truth_k.lon,
      truth_k.h,
      truth_k.vn,
      truth_k.ve,
      truth_k.vd,
      truth_k.cb,
      truth_k.cd,
      lambda,
      beta,
      kappa);
  nav.Propagate(1.5 * conf.meas_dt);
  TrackingStates nav_state = InitTrackingState(
      conf,
      sim.obs,
      conf.init_tow,
      nav.phi_,
      nav.lam_,
      nav.h_,
      nav.vn_,
      nav.ve_,
      nav.vd_,
      nav.cb_,
      nav.cd_,
      lambda,
      beta,
      kappa);
  TrackingStates true_state_kp1 = true_state_k;
  Eigen::VectorXd tR = nav_state.ToW + nav_state.code_psr / navtools::LIGHT_SPEED<>;
  Eigen::VectorXd next_tR(conf.n_sv), next_ToW(conf.n_sv);

  // 8) Initialize output file writers
  FileLoggers log = InitFileLoggers(conf, seed);

  // 9) Run simulation
  Eigen::Matrix3Xcd R1(3, conf.n_sv), R2(3, conf.n_sv), R(3, conf.n_sv);  // correlators
  Eigen::VectorXd dP(conf.n_sv), dR(conf.n_sv), dRR(conf.n_sv), psr_meas(conf.n_sv),
      psrdot_meas(conf.n_sv), dR_var(conf.n_sv), dRR_var(conf.n_sv);

  double T_sim = 2.0 * conf.sim_dt;
  int delta_k_meas = static_cast<int>(conf.meas_dt / conf.sim_dt);
  int delta_k_corr = delta_k_meas / 2;
  int k_corr = 0;
  int k_meas = 0;
  int correlation_half = 1;
  logger->info("Run {}: Starting VT Simulation ({:.3f} s) ...", seed, sw);

  int k = 0;
  while (fid.read(reinterpret_cast<char *>(&truth_kp1), sizeof(Truth) - 16)) {
    // convert degrees to radians
    truth_kp1.lat *= navtools::DEG2RAD<double>;
    truth_kp1.lon *= navtools::DEG2RAD<double>;
    truth_kp1.roll *= navtools::DEG2RAD<double>;
    truth_kp1.pitch *= navtools::DEG2RAD<double>;
    truth_kp1.yaw *= navtools::DEG2RAD<double>;

    // simulate clock errors
    sim.clk.Simulate(truth_kp1.cb, truth_kp1.cd);

    // update true tracking states
    true_state_kp1.ToW.array() += conf.sim_dt;
    lla << truth_kp1.lat, truth_kp1.lon, truth_kp1.h;
    nedv << truth_kp1.vn, truth_kp1.ve, truth_kp1.vd;
    navtools::lla2ecef<double>(xyz, lla);
    navtools::ned2ecefv<double>(xyzv, nedv, lla);
    sim.obs.GetRangeAndRate<false>(
        true_state_kp1.ToW,
        xyz,
        xyzv,
        truth_kp1.cb,
        truth_kp1.cd,
        true_state_kp1.u,
        true_state_kp1.sv_clk,
        true_state_kp1.sv_pos,
        true_state_kp1.sv_vel,
        true_state_kp1.range,
        true_state_kp1.code_psr,
        true_state_kp1.psrdot,
        true_state_kp1.phase_psr);
    true_state_kp1.phase =
        navtools::TWO_PI<> * (conf.intmd_freq * T_sim - true_state_kp1.phase_psr.array() / lambda);
    true_state_kp1.omega =
        navtools::TWO_PI<> * (conf.intmd_freq - true_state_kp1.psrdot.array() / lambda);
    true_state_kp1.chip = true_state_kp1.code_psr / beta;
    true_state_k.chip_rate =
        satutils::GPS_CA_CODE_RATE<> -
        (true_state_kp1.code_psr - true_state_k.code_psr).array() / (conf.sim_dt * beta);

    if (k > 0) {
      // propagate nco
      nav_state.ToW.array() += conf.sim_dt;
      tR.array() += (satutils::GPS_CA_CODE_RATE<> / nav_state.chip_rate.array()) * conf.sim_dt;
      nav_state.chip = (tR - nav_state.ToW) * satutils::GPS_CA_CODE_RATE<>;
      nav_state.phase += nav_state.omega * conf.sim_dt;

      // get true cno
      true_cno = sim.cno.FsplPlusJammerModel(j2s, true_state_k.range);

      // send next sample to correlator sim
      sim.corr.NextSample(
          conf.sim_dt,
          true_cno,
          true_state_k.chip,
          true_state_k.chip_rate,
          true_state_k.phase,
          true_state_k.omega,
          nav_state.chip,
          nav_state.chip_rate,
          nav_state.phase,
          nav_state.omega);

      // correlator update
      if (k_corr == delta_k_corr) {
        switch (correlation_half) {
          case 1:
            sim.corr.GetCorrelators(R1);
            correlation_half = 2;
            break;
          case 2:
            sim.corr.GetCorrelators(R2);
            correlation_half = 1;
            break;
        }
        k_corr = 0;
      }

      // measurement update
      if (k_meas == delta_k_meas) {
        R = R1 + R2;

        // update cno estimate
        est_cno = cno_estimator.Update(R.row(1), conf.meas_dt);

        // calculate discriminators
        CalculateDiscriminators(
            conf.meas_dt, lambda, beta, est_cno, R, R1, R2, dR, dRR, dR_var, dRR_var);

        // navigation update
        psr_meas = (tR - nav_state.ToW) * navtools::LIGHT_SPEED<> + dR;
        psrdot_meas = -lambda * (nav_state.omega.array() / navtools::TWO_PI<> - conf.intmd_freq) +
                      dRR.array();
        nav.GnssUpdate(
            true_state_k.sv_pos, true_state_k.sv_vel, psr_meas, psrdot_meas, dR_var, dRR_var);

        // log results
        Eigen::Vector3d rpy_est, lla_true;
        navtools::quat2euler<double>(rpy_est, nav.q_b_l_, true);
        Eigen::Vector3d lla_est{nav.phi_, nav.lam_, nav.h_};
        lla_true << truth_k.lat, truth_k.lon, truth_k.h;
        Eigen::Vector3d ned_err = navtools::lla2ned<double>(lla_est, lla_true);
        Eigen::Vector3d nedv_err{truth_k.vn - nav.vn_, truth_k.ve - nav.ve_, truth_k.vd - nav.vd_};
        Eigen::Vector3d rpy_err{
            truth_k.roll - rpy_est(0), truth_k.pitch - rpy_est(1), truth_k.yaw - rpy_est(2)};
        double cb_err = truth_k.cb - nav.cb_;
        double cd_err = truth_k.cd - nav.cd_;
        lla_est(0) *= navtools::RAD2DEG<>;
        lla_est(1) *= navtools::RAD2DEG<>;
        rpy_est *= navtools::RAD2DEG<>;
        rpy_err *= navtools::RAD2DEG<>;

        log.navlog.write(reinterpret_cast<char *>(&truth_k.t), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&nav_state.ToW(0)), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&lla_est(0)), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&lla_est(1)), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&lla_est(2)), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&nav.vn_), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&nav.ve_), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&nav.vd_), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&rpy_est(0)), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&rpy_est(1)), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&rpy_est(2)), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&nav.cb_), sizeof(double));
        log.navlog.write(reinterpret_cast<char *>(&nav.cd_), sizeof(double));

        log.errlog.write(reinterpret_cast<char *>(&truth_k.t), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&nav_state.ToW(0)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&ned_err(0)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&ned_err(1)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&ned_err(2)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&nedv_err(0)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&nedv_err(1)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&nedv_err(2)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&rpy_err(0)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&rpy_err(1)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&rpy_err(2)), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&cb_err), sizeof(double));
        log.errlog.write(reinterpret_cast<char *>(&cd_err), sizeof(double));

        log.varlog.write(reinterpret_cast<char *>(&truth_k.t), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav_state.ToW(0)), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(0, 0)), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(1, 1)), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(2, 2)), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(3, 3)), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(4, 4)), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(5, 5)), sizeof(double));
        double tmp = R2Dsq * nav.P_(6, 6);
        log.varlog.write(reinterpret_cast<char *>(&tmp), sizeof(double));
        tmp = R2Dsq * nav.P_(7, 7);
        log.varlog.write(reinterpret_cast<char *>(&tmp), sizeof(double));
        tmp = R2Dsq * nav.P_(8, 8);
        log.varlog.write(reinterpret_cast<char *>(&tmp), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(9, 9)), sizeof(double));
        log.varlog.write(reinterpret_cast<char *>(&nav.P_(10, 10)), sizeof(double));

        for (int i = 0; i < conf.n_sv; i++) {
          // clang-format off
          double true_cno_db = 10.0 * std::log10(true_cno(i));
          double est_cno_db = 10.0 * std::log10(est_cno(i));
          log.channellogs[i].write(reinterpret_cast<char *>(&truth_k.t), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.ToW(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k.phase(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k.omega(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k.chip(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k.chip_rate(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&true_cno_db), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.phase(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.omega(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.chip(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.chip_rate(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&est_cno_db), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&R(0, i)), sizeof(std::complex<double>));
          log.channellogs[i].write(reinterpret_cast<char *>(&R(1, i)), sizeof(std::complex<double>));
          log.channellogs[i].write(reinterpret_cast<char *>(&R(2, i)), sizeof(std::complex<double>));
          log.channellogs[i].write(reinterpret_cast<char *>(&R1(1, i)), sizeof(std::complex<double>));
          log.channellogs[i].write(reinterpret_cast<char *>(&R2(1, i)), sizeof(std::complex<double>));
          log.channellogs[i].write(reinterpret_cast<char *>(&dP(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&dRR(i)), sizeof(double));
          log.channellogs[i].write(reinterpret_cast<char *>(&dR(i)), sizeof(double));
          // clang-format on
        }

        // navigation propagation
        nav.Propagate(conf.meas_dt);
        lla << nav.phi_, nav.lam_, nav.h_;
        nedv << nav.vn_, nav.ve_, nav.vd_;
        navtools::lla2ecef<double>(xyz, lla);
        navtools::ned2ecefv<double>(xyzv, nedv, lla);
        next_ToW = nav_state.ToW.array() + conf.meas_dt;

        // nco predictions
        sim.obs.GetRangeAndRate<false>(
            next_ToW,
            xyz,
            xyzv,
            nav.cb_,
            nav.cd_,
            nav_state.u,
            nav_state.sv_clk,
            nav_state.sv_pos,
            nav_state.sv_vel,
            nav_state.range,
            nav_state.code_psr,
            nav_state.psrdot,
            nav_state.phase_psr);
        next_tR = next_ToW + nav_state.code_psr /
                                 navtools::LIGHT_SPEED<>;  //- nav_state.sv_clk.row(0).transpose()
        nav_state.chip_rate =
            (satutils::GPS_CA_CODE_RATE<> * conf.meas_dt) / (next_tR - tR).array();
        nav_state.omega =
            navtools::TWO_PI<> * (conf.intmd_freq - nav_state.psrdot.array() / lambda);

        k_meas = 0;
        // last_meas_t = T_sim;
      }

      k_meas++;
      k_corr++;
    }

    // increment
    truth_k = truth_kp1;
    true_state_k = true_state_kp1;
    T_sim += conf.sim_dt;
    k++;
  }

  fid.close();
  log.navlog.close();
  log.errlog.close();
  log.varlog.close();
  for (int i = 0; i < conf.n_sv; i++) {
    log.channellogs[i].close();
  }
  // log.navlog->flush();
  // log.errlog->flush();
  // log.varlog->flush();
  // for (int i = 0; i < conf.n_sv; i++) {
  //   log.channellogs[i]->flush();
  // }
  logger->info("Run {}: Done ({:.3f} s) ...", seed, sw);
};

//! ================================================================================================

void vt_array_sim(
    SimParam &conf, spdlog::stopwatch &sw, std::shared_ptr<spdlog::logger> logger, int seed) {
  // constants
  double lambda = navtools::LIGHT_SPEED<double> / satutils::GPS_L1_FREQUENCY<double>;
  double beta = navtools::LIGHT_SPEED<double> / satutils::GPS_CA_CODE_RATE<double>;
  double kappa = satutils::GPS_CA_CODE_RATE<double> / satutils::GPS_L1_FREQUENCY<double>;

  // 2) Parse satellite ephemeris
  std::vector<satutils::KeplerElements<double>> elem;
  std::vector<satutils::KlobucharElements<double>> klob;
  ParseEphem(conf, elem, klob);

  // 3) Skip to desired time in truth data
  Truth truth_k, truth_kp1;
  std::ifstream fid = OpenDataFile(conf, truth_k);
  Eigen::Vector3d rpy_est, lla_est, nedv_est, xyz_est, xyzv_est, rpy_true, lla_true, nedv_true,
      xyz_true, xyzv_true;

  // 4) Create desired random noise generator
  std::mt19937_64 noise_gen;
  noise_gen.seed(seed);
  std::normal_distribution<double> noise_dist(0.0, 1.0);

  // 5) Initialize simulators (4 different correlator simulators)
  ArraySimulators sim = CreateArraySimulators(conf, elem, klob, lambda, noise_gen, noise_dist);
  sim.clk.GetCurrentState(truth_k.cb, truth_k.cd);

  // 6) initialize navigator/estimators (2 different C/No estimators)
  Eigen::VectorXd true_cno(conf.n_sv), est_cno(conf.n_sv), est_cno_bs(conf.n_sv),
      j2s{std::pow(10.0, conf.j2s / 10.0) * Eigen::VectorXd::Ones(conf.n_sv)};
  std::vector<CnoEstimator> cno_estimators{CnoEstimator(conf.n_sv), CnoEstimator(conf.n_sv)};
  sturdins::Kns nav = InitNavigator(conf, truth_k, noise_gen, noise_dist);

  // 7.1) Initialize tracking loop states
  rpy_true << truth_k.roll, truth_k.pitch, truth_k.yaw;
  lla_true << truth_k.lat, truth_k.lon, truth_k.h;
  navtools::lla2ecef<double>(xyz_true, lla_true);
  Eigen::Matrix3d C_b_n = navtools::euler2dcm<double>(rpy_true, true);
  Eigen::Matrix3d C_n_e = navtools::ned2ecefDcm<double>(lla_true);
  Eigen::MatrixXd ant_xyz_ecef = (C_n_e * (C_b_n * conf.ant_xyz)).colwise() + xyz_true;
  Eigen::MatrixXd ant_lla(3, conf.n_ant);
  std::vector<TrackingStates> true_state_k;
  for (int i = 0; i < conf.n_ant; i++) {
    navtools::ecef2lla<double>(ant_lla.col(i), ant_xyz_ecef.col(i));
    true_state_k.push_back(InitTrackingState(
        conf,
        sim.obs,
        conf.init_tow,
        ant_lla(0, i),
        ant_lla(1, i),
        ant_lla(2, i),
        truth_k.vn,
        truth_k.ve,
        truth_k.vd,
        truth_k.cb,
        truth_k.cd,
        lambda,
        beta,
        kappa));
  }
  nav.Propagate(1.5 * conf.meas_dt);
  TrackingStates nav_state = InitTrackingState(
      conf,
      sim.obs,
      conf.init_tow,
      nav.phi_,
      nav.lam_,
      nav.h_,
      nav.vn_,
      nav.ve_,
      nav.vd_,
      nav.cb_,
      nav.cd_,
      lambda,
      beta,
      kappa);
  std::vector<TrackingStates> true_state_kp1 = true_state_k;
  Eigen::VectorXd tR = nav_state.ToW + nav_state.code_psr / navtools::LIGHT_SPEED<>;
  Eigen::VectorXd next_tR(conf.n_sv), next_ToW(conf.n_sv);

  // 7.2) Initialize beam steering weights
  Eigen::MatrixXcd W_bs(conf.n_ant, conf.n_sv), est_spatial_phase(conf.n_ant, conf.n_sv);
  Eigen::MatrixXd est_u(3, conf.n_sv);
  Eigen::Matrix3d C_n_b = C_b_n.transpose(), C_e_n = C_n_e.transpose();
  for (int i = 0; i < conf.n_sv; i++) {
    est_u.col(i) = C_n_b * (C_e_n * nav_state.u.col(i));
    est_spatial_phase.col(i) =
        navtools::TWO_PI<> / lambda * (conf.ant_xyz.transpose() * est_u.col(i));
    W_bs.col(i) = (navtools::COMPLEX_I<> * est_spatial_phase.col(i)).array().exp();
  }

  // 8) Initialize output file writers
  FileLoggers log = InitFileLoggers(conf, seed);

  // 9) Run simulation
  Eigen::Matrix3Xcd R1_bs(3, conf.n_sv), R2_bs(3, conf.n_sv), R_bs(3, conf.n_sv);  // beamsteer
  std::vector<Eigen::Matrix3Xcd> R1, R2, R;                                        // correlators
  for (int i = 0; i < conf.n_ant; i++) {
    Eigen::Matrix3Xcd tmp{Eigen::Matrix3Xcd::Zero(3, conf.n_sv)};
    R1.push_back(tmp);
    R2.push_back(tmp);
    R.push_back(tmp);
  }
  Eigen::VectorXd dR(conf.n_sv), dRR(conf.n_sv), psr_meas(conf.n_sv), psrdot_meas(conf.n_sv),
      dR_var(conf.n_sv), dRR_var(conf.n_sv);
  Eigen::MatrixXd dP(conf.n_ant, conf.n_sv), dP_var(conf.n_ant, conf.n_sv);
  Eigen::MatrixXcd RP(conf.n_ant, conf.n_sv);

  double T_sim = 2.0 * conf.sim_dt;
  int delta_k_meas = static_cast<int>(conf.meas_dt / conf.sim_dt);
  int delta_k_corr = delta_k_meas / 2;
  int k_corr = 0;
  int k_meas = 0;
  int correlation_half = 1;
  logger->info("Run {}: Starting Multi-Antenna VT Simulation ({:.3f} s) ...", seed, sw);

  int k = 0;
  // while (fid.read(reinterpret_cast<char *>(&truth_kp1), sizeof(Truth) - 16)) {
  for (int i = 0; i < 10; i++) {
    fid.read(reinterpret_cast<char *>(&truth_kp1), sizeof(Truth) - 16);

    // convert degrees to radians
    truth_kp1.lat *= navtools::DEG2RAD<double>;
    truth_kp1.lon *= navtools::DEG2RAD<double>;
    truth_kp1.roll *= navtools::DEG2RAD<double>;
    truth_kp1.pitch *= navtools::DEG2RAD<double>;
    truth_kp1.yaw *= navtools::DEG2RAD<double>;

    // simulate clock errors
    sim.clk.Simulate(truth_kp1.cb, truth_kp1.cd);

    // update true tracking states
    lla_true << truth_kp1.lat, truth_kp1.lon, truth_kp1.h;
    nedv_true << truth_kp1.vn, truth_kp1.ve, truth_kp1.vd;
    rpy_true << truth_kp1.roll, truth_kp1.pitch, truth_kp1.yaw;
    navtools::lla2ecef<double>(xyz_true, lla_true);
    navtools::euler2dcm<double>(C_b_n, rpy_true, true);
    navtools::ned2ecefDcm<double>(C_n_e, lla_true);
    ant_xyz_ecef = (C_n_e * (C_b_n * conf.ant_xyz)).colwise() + xyz_true;
    xyzv_true = C_n_e * nedv_true;
    for (int i = 0; i < conf.n_ant; i++) {
      true_state_kp1[i].ToW.array() += conf.sim_dt;
      sim.obs.GetRangeAndRate<false>(
          true_state_kp1[i].ToW,
          ant_xyz_ecef.col(i),
          xyzv_true,
          truth_kp1.cb,
          truth_kp1.cd,
          true_state_kp1[i].u,
          true_state_kp1[i].sv_clk,
          true_state_kp1[i].sv_pos,
          true_state_kp1[i].sv_vel,
          true_state_kp1[i].range,
          true_state_kp1[i].code_psr,
          true_state_kp1[i].psrdot,
          true_state_kp1[i].phase_psr);
      true_state_kp1[i].phase = navtools::TWO_PI<> * (conf.intmd_freq * T_sim -
                                                      true_state_kp1[i].phase_psr.array() / lambda);
      true_state_kp1[i].omega =
          navtools::TWO_PI<> * (conf.intmd_freq - true_state_kp1[i].psrdot.array() / lambda);
      true_state_kp1[i].chip = true_state_kp1[i].code_psr / beta;
      true_state_k[i].chip_rate =
          satutils::GPS_CA_CODE_RATE<> -
          (true_state_kp1[i].code_psr - true_state_k[i].code_psr).array() / (conf.sim_dt * beta);
    }

    if (k > 0) {
      // propagate nco
      nav_state.ToW.array() += conf.sim_dt;
      tR.array() += (satutils::GPS_CA_CODE_RATE<> / nav_state.chip_rate.array()) * conf.sim_dt;
      nav_state.chip = (tR - nav_state.ToW) * satutils::GPS_CA_CODE_RATE<>;
      nav_state.phase += nav_state.omega * conf.sim_dt;
      // nav_state.phase = true_state_k[0].phase;

      // get true cno
      true_cno = sim.cno.FsplPlusJammerModel(j2s, true_state_k[0].range);

      // send next sample to correlator sim
      for (int i = 0; i < conf.n_ant; i++) {
        sim.corr[i].NextSample(
            conf.sim_dt,
            true_cno,
            true_state_k[i].chip,
            true_state_k[i].chip_rate,
            true_state_k[i].phase,
            true_state_k[i].omega,
            nav_state.chip,
            nav_state.chip_rate,
            nav_state.phase,
            nav_state.omega);
      }

      // correlator update
      if (k_corr == delta_k_corr) {
        for (int i = 0; i < conf.n_ant; i++) {
          switch (correlation_half) {
            case 1:
              sim.corr[i].GetCorrelators(R1[i]);
              // correlation_half = 2;
              break;
            case 2:
              sim.corr[i].GetCorrelators(R2[i]);
              // correlation_half = 1;
              break;
          }
        }
        correlation_half = (correlation_half == 1) ? 2 : 1;
        k_corr = 0;
      }

      // measurement update
      if (k_meas == delta_k_meas) {
        for (int i = 0; i < conf.n_ant; i++) {
          R[i] = R1[i] + R2[i];
          RP.row(i) = R[i].row(1);
        }

        // post-correlation beamsteering
        // R_bs - 3 x n_sv
        // R    - 4 x (3 x n_sv)
        // W_bs - 4 x n_sv
        R_bs.setZero();
        R1_bs.setZero();
        R2_bs.setZero();
        for (int i = 0; i < conf.n_ant; i++) {
          for (int j = 0; j < conf.n_sv; j++) {
            R_bs.col(j) += W_bs(i, j) * R[i].col(j);
            R1_bs.col(j) += W_bs(i, j) * R1[i].col(j);
            R2_bs.col(j) += W_bs(i, j) * R2[i].col(j);
          }
        }

        // update cno estimate
        est_cno_bs = cno_estimators[0].Update(R_bs.row(1), conf.meas_dt);
        est_cno = cno_estimators[1].Update(R[0].row(1), conf.meas_dt);
        std::cout << "est_cno: " << est_cno.transpose() << "\n";

        // calculate discriminators
        // CalculateDiscriminators(
        //     conf.meas_dt, lambda, beta, est_cno, R[0], R1[0], R2[0], dR, dRR, dR_var, dRR_var);
        CalculateArrayDiscriminators(
            conf.meas_dt,
            lambda,
            beta,
            est_cno,
            est_cno_bs,
            RP,
            R_bs,
            R1_bs,
            R2_bs,
            dR,
            dRR,
            dP,
            dR_var,
            dRR_var,
            dP_var);

        // navigation update
        psr_meas = (tR - nav_state.ToW) * navtools::LIGHT_SPEED<> + dR;
        psrdot_meas = -lambda * (nav_state.omega.array() / navtools::TWO_PI<> - conf.intmd_freq) +
                      dRR.array();
        // nav.GnssUpdate(
        //     true_state_k[0].sv_pos, true_state_k[0].sv_vel, psr_meas, psrdot_meas, dR_var,
        //     dRR_var);
        nav.PhasedArrayUpdate(
            true_state_k[0].sv_pos,
            true_state_k[0].sv_vel,
            psr_meas,
            psrdot_meas,
            dP,
            dR_var,
            dRR_var,
            dP_var,
            conf.ant_xyz,
            conf.n_ant,
            lambda / navtools::TWO_PI<>);

        // log results
        LogResult(
            conf,
            log,
            truth_k,
            nav,
            true_state_k,
            nav_state,
            true_cno,
            est_cno,
            est_cno_bs,
            dR,
            dRR,
            dP,
            R_bs,
            R1_bs,
            R2_bs,
            RP);

        // navigation propagation
        nav.Propagate(conf.meas_dt);
        lla_est << nav.phi_, nav.lam_, nav.h_;
        nedv_est << nav.vn_, nav.ve_, nav.vd_;
        navtools::quat2euler<double>(rpy_est, nav.q_b_l_, true);
        navtools::lla2ecef<double>(xyz_est, lla_est);
        navtools::ned2ecefv<double>(xyzv_est, nedv_est, lla_est);
        next_ToW = nav_state.ToW.array() + conf.meas_dt;

        // nco predictions
        sim.obs.GetRangeAndRate<false>(
            next_ToW,
            xyz_est,
            xyzv_est,
            nav.cb_,
            nav.cd_,
            nav_state.u,
            nav_state.sv_clk,
            nav_state.sv_pos,
            nav_state.sv_vel,
            nav_state.range,
            nav_state.code_psr,
            nav_state.psrdot,
            nav_state.phase_psr);
        next_tR = next_ToW + nav_state.code_psr /
                                 navtools::LIGHT_SPEED<>;  //- nav_state.sv_clk.row(0).transpose()
        nav_state.chip_rate =
            (satutils::GPS_CA_CODE_RATE<> * conf.meas_dt) / (next_tR - tR).array();
        nav_state.omega =
            navtools::TWO_PI<> * (conf.intmd_freq - nav_state.psrdot.array() / lambda);

        // update beamsteering weights
        navtools::euler2dcm<double>(C_b_n, rpy_est, true);
        navtools::ned2ecefDcm<double>(C_n_e, lla_est);
        C_n_b = C_b_n.transpose();
        C_e_n = C_n_e.transpose();
        for (int i = 0; i < conf.n_sv; i++) {
          est_u.col(i) = C_n_b * (C_e_n * nav_state.u.col(i));
          est_spatial_phase.col(i) =
              navtools::TWO_PI<> / lambda * (conf.ant_xyz.transpose() * est_u.col(i));
          W_bs.col(i) = (navtools::COMPLEX_I<> * est_spatial_phase.col(i)).array().exp();
        }

        k_meas = 0;
      }

      k_meas++;
      k_corr++;
    }

    // increment
    truth_k = truth_kp1;
    true_state_k = true_state_kp1;
    T_sim += conf.sim_dt;
    k++;
  }

  fid.close();
  log.navlog.close();
  log.errlog.close();
  log.varlog.close();
  for (int i = 0; i < conf.n_sv; i++) {
    log.channellogs[i].close();
  }
  // log.navlog->flush();
  // spdlog::drop("nav-logger" + std::to_string(seed));
  // log.errlog->flush();
  // spdlog::drop("err-logger" + std::to_string(seed));
  // log.varlog->flush();
  // spdlog::drop("var-logger" + std::to_string(seed));
  // for (int i = 0; i < conf.n_sv; i++) {
  //   log.channellogs[i]->flush();
  //   spdlog::drop("channel-" + std::to_string(i) + "-logger" + std::to_string(seed));
  // }
  logger->info("Run {}: Done ({:.3f} s) ...", seed, sw);
}

//! ================================================================================================

int main(int argc, char *argv[]) {
  std::cout << std::setprecision(11);
  std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("console");
  logger->set_pattern("\033[1;34m[%D %T.%e][%^%l%$\033[1;34m]: \033[0m%v");
  spdlog::stopwatch sw;
  logger->info("Initializing ({:.3f} s) ...", sw);

  // 1) Parse yaml configuration
  SimParam conf = ParseConfig(argc, argv);

  // RUN SIMULATOR
  vt_array_sim(conf, sw, logger, 1);
  // std::function<void(SimParam &, spdlog::stopwatch &, std::shared_ptr<spdlog::logger>, int)>
  // func; if (conf.is_multi_antenna) {
  //   func = &vt_array_sim;
  // } else {
  //   func = &vt_sim;
  // }
  // // std::cout << std::thread::hardware_concurrency() << "\n";
  // int n_threads = std::thread::hardware_concurrency() / 2;
  // int n_loops = conf.n_runs / n_threads;

  // for (int i = 0; i < n_loops; i++) {
  //   std::vector<std::future<void>> futures;
  //   for (int j = 0; j < n_threads; j++) {
  //     // std::cout << 100 * ((i * n_loops) + j + 1) << "\n";
  //     futures.push_back(std::async(
  //         std::launch::async, func, std::ref(conf), std::ref(sw), logger, (i * n_threads) + j +
  //         1));
  //   }
  //   for (std::future<void> &f : futures) {
  //     f.get();
  //   }
  // }
  // logger->info("Monte Carlo Done ({:.3f} s) ...", sw);

  return 0;
}