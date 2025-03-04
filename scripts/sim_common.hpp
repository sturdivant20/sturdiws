
// #include <spdlog/sinks/basic_file_sink.h>
// #include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <navsim/clock-sim.hpp>
#include <navsim/cno-sim.hpp>
#include <navsim/correlator-sim.hpp>
#include <navsim/observable-sim.hpp>
#include <navtools/attitude.hpp>
#include <navtools/constants.hpp>
#include <navtools/math.hpp>
#include <random>
#include <satutils/atmosphere.hpp>
#include <satutils/ephemeris.hpp>
#include <string>
#include <sturdins/kns.hpp>
#include <sturdins/nav-clock.hpp>
#include <sturdio/io-tools.hpp>
#include <sturdio/yaml-parser.hpp>
#include <vector>

struct SimParam {
  std::string scenario;
  std::string clock_model;
  double vel_process_psd;
  double att_process_psd;
  double sec_to_skip;
  double init_tow;
  double intmd_freq;
  double sim_dt;
  double corr_dt;
  double meas_dt;
  double tap_epl_wide;
  double tap_epl;
  double tap_epl_narrow;
  double init_cb;
  double init_cd;
  Eigen::MatrixXd init_cov;
  std::string jammer_modulation;
  std::string jammer_type;
  double j2s;
  bool add_init_err;
  bool is_multi_antenna;
  int n_ant;
  Eigen::MatrixXd ant_xyz;
  std::string data_file;
  std::string ephem_file;
  std::string out_folder;
  int n_runs;
  int n_sv;
};

struct Truth {
  double t;
  double lat;
  double lon;
  double h;
  double vn;
  double ve;
  double vd;
  double roll;
  double pitch;
  double yaw;
  double cb;
  double cd;
};

struct TrackingStates {
  Eigen::VectorXd ToW;
  Eigen::VectorXd range;
  Eigen::VectorXd code_psr;
  Eigen::VectorXd phase_psr;
  Eigen::VectorXd psrdot;
  Eigen::VectorXd chip;
  Eigen::VectorXd chip_rate;
  Eigen::VectorXd phase;
  Eigen::VectorXd omega;
  Eigen::MatrixXd sv_pos;
  Eigen::MatrixXd sv_vel;
  Eigen::MatrixXd sv_clk;
  Eigen::MatrixXd u;
};

struct Simulators {
  navsim::ObservablesSim<double> obs;
  navsim::CorrelatorSim<double> corr;
  navsim::ClockSim<double> clk;
  navsim::CnoSim<double> cno;
};
struct ArraySimulators {
  navsim::ObservablesSim<double> obs;
  std::vector<navsim::CorrelatorSim<double>> corr;
  navsim::ClockSim<double> clk;
  navsim::CnoSim<double> cno;
};

struct FileLoggers {
  // std::shared_ptr<spdlog::logger> navlog;
  // std::shared_ptr<spdlog::logger> errlog;
  // std::shared_ptr<spdlog::logger> varlog;
  // std::vector<std::shared_ptr<spdlog::logger>> channellogs;
  std::ofstream navlog;
  std::ofstream errlog;
  std::ofstream varlog;
  std::vector<std::ofstream> channellogs;
};

//! ------------------------------------------------------------------------------------------------

// *=== CNoEstimator ===*
class CnoEstimator {
 private:
  Eigen::VectorXcd prev_P_;
  // Eigen::VectorXd pc_;
  // Eigen::VectorXd pn_;
  Eigen::VectorXd m2_;
  Eigen::VectorXd m4_;
  double alpha_;

 public:
  CnoEstimator(int n_sv, double alpha = 0.005)
      : prev_P_{Eigen::VectorXcd::Zero(n_sv)},
        m2_{Eigen::VectorXd::Zero(n_sv)},
        m4_{Eigen::VectorXd::Zero(n_sv)},
        alpha_{alpha} {};
  ~CnoEstimator() = default;

  Eigen::VectorXd Update(const Eigen::Ref<const Eigen::VectorXcd> &P, const double &T) {
    // Calculate powers
    // if (prev_P_.sum() == 0.0) {
    //   prev_P_ = P;
    //   return 1000.0 * Eigen::VectorXd::Ones(P.size());
    // } else if (pc_.sum() == 0.0) {
    //   Eigen::VectorXd P_old = prev_P_.cwiseAbs2();
    //   Eigen::VectorXd P_new = P.cwiseAbs2();
    //   pc_ = 0.5 * (P_old + P_new);
    //   pn_ = (P_new.cwiseSqrt() - P_old.cwiseSqrt()).array().pow(2);
    // } else {
    //   Eigen::VectorXd P_old = prev_P_.cwiseAbs2();
    //   Eigen::VectorXd P_new = P.cwiseAbs2();
    //   Eigen::VectorXd P_carrier = 0.5 * (P_old + P_new);
    //   Eigen::VectorXd P_noise = (P_new.cwiseSqrt() - P_old.cwiseSqrt()).array().pow(2);
    //   pc_.array() += alpha_ * (P_carrier - pc_).array();
    //   pn_.array() += alpha_ * (P_noise - pn_).array();
    // }
    // prev_P_ = P;
    // return pc_.array() / (pn_.array() * T);  // (1 / (PN / PC)) / T
    if (prev_P_.sum() == 0.0) {
      prev_P_ = P;
      return 1000.0 * Eigen::VectorXd::Ones(P.size());
    } else if (m2_.sum() == 0.0) {
      Eigen::VectorXd p_old = prev_P_.cwiseAbs2();
      Eigen::VectorXd p_new = P.cwiseAbs2();
      m2_ = 0.5 * (p_old + p_new);
      m4_ = 0.5 * (p_old * p_old + p_new * p_new);
    } else {
      Eigen::VectorXd p_new = P.cwiseAbs2();
      m2_.array() += alpha_ * (p_new - m2_).array();
      m4_.array() += alpha_ * (p_new * p_new - m4_).array();
    }

    prev_P_ = P;
    Eigen::VectorXd pc = (2.0 * m2_ * m2_ - m4_).cwiseSqrt();
    std::cout << "m2: " << m2_.transpose() << "\n";
    std::cout << "m4: " << m4_.transpose() << "\n";
    std::cout << "pc: " << pc.transpose() << "\n";
    return (pc.array() / (m2_ - pc).array()) / T;
  }
};

//! ------------------------------------------------------------------------------------------------

// *=== ParseConfig ===*
inline SimParam ParseConfig(int argc, char *argv[]) {
  std::string yaml_filename;
  if (argc > 1) {
    yaml_filename = argv[1];
  } else {
    yaml_filename = "config/vt_correlator_sim.yaml";
  }
  sturdio::YamlParser yp(yaml_filename);
  SimParam conf{
      yp.GetVar<std::string>("scenario"),
      yp.GetVar<std::string>("clock_model"),
      yp.GetVar<double>("vel_process_psd"),
      yp.GetVar<double>("att_process_psd"),
      yp.GetVar<double>("sec_to_skip"),
      yp.GetVar<double>("init_tow"),
      yp.GetVar<double>("intmd_freq"),
      yp.GetVar<double>("sim_dt"),
      0.5 * yp.GetVar<double>("meas_dt"),
      yp.GetVar<double>("meas_dt"),
      yp.GetVar<double>("tap_epl_wide"),
      yp.GetVar<double>("tap_epl"),
      yp.GetVar<double>("tap_epl_narrow"),
      yp.GetVar<double>("init_cb"),
      yp.GetVar<double>("init_cd"),
      Eigen::MatrixXd::Zero(11, 11),
      yp.GetVar<std::string>("jammer_modulation"),
      yp.GetVar<std::string>("jammer_type"),
      yp.GetVar<double>("j2s"),
      yp.GetVar<bool>("add_init_err"),
      yp.GetVar<bool>("is_multi_antenna"),
      yp.GetVar<int>("n_ant"),
      Eigen::MatrixXd::Zero(3, yp.GetVar<int>("n_ant")),
      yp.GetVar<std::string>("data_file"),
      yp.GetVar<std::string>("ephem_file"),
      yp.GetVar<std::string>("out_folder"),
      yp.GetVar<int>("n_runs"),
      0};
  if (conf.n_ant > 1) {
    std::vector<double> vec;
    std::string item;
    for (int i = 0; i < conf.n_ant; i++) {
      item = "ant_xyz_" + std::to_string(i);
      yp.GetVar<std::vector<double>>(vec, item);
      conf.ant_xyz.col(i) = Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
      vec.clear();
    }
  }
  std::vector<double> vec;
  yp.GetVar<std::vector<double>>(vec, "init_cov");
  conf.init_cov.diagonal() = Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
  return conf;
};

//! ------------------------------------------------------------------------------------------------

// *=== ParseEphem ===*
inline void ParseEphem(
    SimParam &conf,
    std::vector<satutils::KeplerElements<double>> &elem,
    std::vector<satutils::KlobucharElements<double>> &klob) {
  satutils::KlobucharElements<double> tmp1{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  satutils::KeplerElements<double> tmp2;
  std::ifstream fid(conf.ephem_file, std::ios::binary);
  if (!fid) {
    std::cerr << "Error opening file!\n";
  }
  while (fid.read(reinterpret_cast<char *>(&tmp2), sizeof(satutils::KeplerElements<double>))) {
    // std::cout << "Next ephemerides: ";
    // std::cout << "\n\tiode     : " << tmp.iode << "\n\tiodc     : " << tmp.iodc
    //           << "\n\ttoe      : " << tmp.toe << "\n\ttoc      : " << tmp.toc
    //           << "\n\ttgd      : " << tmp.tgd << "\n\toaf2     : " << tmp.af2
    //           << "\n\taf1      : " << tmp.af1 << "\n\taf0      : " << tmp.af0
    //           << "\n\te        : " << tmp.e << "\n\tsqrtA    : " << tmp.sqrtA
    //           << "\n\tdeltan   : " << tmp.deltan << "\n\tm0       : " << tmp.m0
    //           << "\n\tomega0   : " << tmp.omega0 << "\n\tomega    : " << tmp.omega
    //           << "\n\tomegaDot : " << tmp.omegaDot << "\n\ti0       : " << tmp.i0
    //           << "\n\tiDot     : " << tmp.iDot << "\n\tcuc      : " << tmp.cuc
    //           << "\n\tcus      : " << tmp.cus << "\n\tcic      : " << tmp.cic
    //           << "\n\tcis      : " << tmp.cis << "\n\tcrc      : " << tmp.crc
    //           << "\n\tcrs      : " << tmp.crs << "\n\tura      : " << tmp.ura
    //           << "\n\thealth   : " << tmp.health << "\n\n";
    elem.push_back(tmp2);
    klob.push_back(tmp1);
  }
  fid.close();
  conf.n_sv = elem.size();
};

//! ------------------------------------------------------------------------------------------------

// *=== OpenDataFile ===*
inline std::ifstream OpenDataFile(SimParam &conf, Truth &truth) {
  std::ifstream fid(conf.data_file, std::ios::binary);
  truth.t = 0.0;
  while (truth.t < conf.sec_to_skip) {
    fid.read(reinterpret_cast<char *>(&truth), sizeof(Truth) - 16);
  }
  truth.lat *= navtools::DEG2RAD<double>;
  truth.lon *= navtools::DEG2RAD<double>;
  truth.roll *= navtools::DEG2RAD<double>;
  truth.pitch *= navtools::DEG2RAD<double>;
  truth.yaw *= navtools::DEG2RAD<double>;
  return fid;
};

//! ------------------------------------------------------------------------------------------------

// *=== CreateSimulators ===*
inline Simulators CreateSimulators(
    SimParam &conf,
    std::vector<satutils::KeplerElements<double>> &elem,
    std::vector<satutils::KlobucharElements<double>> &klob,
    double &lambda,
    std::default_random_engine noise_gen,
    std::normal_distribution<double> noise_dist) {
  sturdins::NavigationClock clk_psd = sturdins::GetNavClock(conf.clock_model);
  return Simulators{
      navsim::ObservablesSim<double>(elem, klob),
      navsim::CorrelatorSim<double>(conf.tap_epl, conf.n_sv, 2, noise_gen, noise_dist),
      navsim::ClockSim<double>(
          clk_psd.h0,
          clk_psd.h1,
          clk_psd.h2,
          conf.init_cb,
          conf.init_cd,
          conf.sim_dt,
          conf.add_init_err,
          noise_gen,
          noise_dist),
      navsim::CnoSim<double>(
          conf.n_sv,
          lambda,
          satutils::GPS_CA_CODE_RATE<double>,
          conf.jammer_modulation,
          conf.jammer_type)};
};

// *=== CreateSimulators ===*
inline ArraySimulators CreateArraySimulators(
    SimParam &conf,
    std::vector<satutils::KeplerElements<double>> &elem,
    std::vector<satutils::KlobucharElements<double>> &klob,
    double &lambda,
    std::default_random_engine noise_gen,
    std::normal_distribution<double> noise_dist) {
  sturdins::NavigationClock clk_psd = sturdins::GetNavClock(conf.clock_model);

  std::vector<navsim::CorrelatorSim<double>> tmp1;
  for (int i = 0; i < conf.n_ant; i++) {
    tmp1.push_back(
        navsim::CorrelatorSim<double>(conf.tap_epl, conf.n_sv, 2, noise_gen, noise_dist));
  }
  return ArraySimulators{
      navsim::ObservablesSim<double>(elem, klob),
      tmp1,
      navsim::ClockSim<double>(
          clk_psd.h0,
          clk_psd.h1,
          clk_psd.h2,
          conf.init_cb,
          conf.init_cd,
          conf.sim_dt,
          conf.add_init_err,
          noise_gen,
          noise_dist),
      navsim::CnoSim<double>(
          conf.n_sv,
          lambda,
          satutils::GPS_CA_CODE_RATE<double>,
          conf.jammer_modulation,
          conf.jammer_type)};
};

//! ------------------------------------------------------------------------------------------------

// *=== InitNavigator ===*
inline sturdins::Kns InitNavigator(
    SimParam &conf,
    Truth &truth,
    std::default_random_engine noise_gen,
    std::normal_distribution<double> noise_dist) {
  Eigen::Vector3d lla, nedv, rpy;

  if (conf.add_init_err) {
    // include initial errors
    Eigen::Vector3d ned, lla2;
    lla2 << truth.lat, truth.lon, truth.h;
    ned << std::sqrt(conf.init_cov(0, 0)) * noise_dist(noise_gen),
        std::sqrt(conf.init_cov(1, 1)) * noise_dist(noise_gen),
        std::sqrt(conf.init_cov(2, 2)) * noise_dist(noise_gen);
    navtools::ned2lla<double>(lla, ned, lla2);
    nedv << truth.vn + std::sqrt(conf.init_cov(3, 3)) * noise_dist(noise_gen),
        truth.ve + std::sqrt(conf.init_cov(4, 4)) * noise_dist(noise_gen),
        truth.vd + std::sqrt(conf.init_cov(5, 5)) * noise_dist(noise_gen);
    rpy << truth.roll + std::sqrt(conf.init_cov(6, 6)) * noise_dist(noise_gen),
        truth.pitch + std::sqrt(conf.init_cov(7, 7)) * noise_dist(noise_gen),
        truth.yaw + std::sqrt(conf.init_cov(8, 8)) * noise_dist(noise_gen);
  } else {
    // no initial errors
    lla << truth.lat, truth.lon, truth.h;
    nedv << truth.vn, truth.ve, truth.vd;
    rpy << truth.roll, truth.pitch, truth.yaw;
  }

  sturdins::NavigationClock clk_psd = sturdins::GetNavClock(conf.clock_model);
  sturdins::Kns nav;
  nav.P_ = conf.init_cov;
  nav.SetPosition(lla(0), lla(1), lla(2));
  nav.SetVelocity(nedv(0), nedv(1), nedv(2));
  nav.SetAttitude(rpy(0), rpy(1), rpy(2));
  nav.SetClock(conf.init_cb, conf.init_cd);
  nav.SetClockSpec(clk_psd.h0, clk_psd.h1, clk_psd.h2);
  nav.SetProcessNoise(conf.vel_process_psd, conf.att_process_psd);

  return nav;
};

//! ------------------------------------------------------------------------------------------------

// *=== InitTrackingState ===*
inline TrackingStates InitTrackingState(
    SimParam &conf,
    navsim::ObservablesSim<double> &sim,
    const double &ToW,
    const double &lat,
    const double &lon,
    const double &h,
    const double &vn,
    const double &ve,
    const double &vd,
    const double &cb,
    const double &cd,
    const double &lambda,
    const double &beta,
    const double &kappa) {
  TrackingStates state{
      Eigen::VectorXd::Ones(conf.n_sv) * conf.init_tow,
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::VectorXd::Zero(conf.n_sv),
      Eigen::MatrixXd::Zero(3, conf.n_sv),
      Eigen::MatrixXd::Zero(3, conf.n_sv),
      Eigen::MatrixXd::Zero(3, conf.n_sv),
      Eigen::MatrixXd::Zero(3, conf.n_sv)};

  Eigen::Vector3d lla{lat, lon, h};
  Eigen::Vector3d nedv{vn, ve, vd};
  Eigen::Vector3d xyz, xyzv;
  navtools::lla2ecef<double>(xyz, lla);
  navtools::ned2ecefv<double>(xyzv, nedv, lla);
  sim.GetRangeAndRate<false>(
      state.ToW,
      xyz,
      xyzv,
      cb,
      cd,
      state.u,
      state.sv_clk,
      state.sv_pos,
      state.sv_vel,
      state.range,
      state.code_psr,
      state.psrdot,
      state.phase_psr);
  double dt = ToW - conf.init_tow;
  state.phase = navtools::TWO_PI<> * (conf.intmd_freq * dt - state.phase_psr.array() / lambda);
  state.omega = navtools::TWO_PI<> * (conf.intmd_freq - state.psrdot.array() / lambda);
  state.chip = state.code_psr / beta;
  state.chip_rate = satutils::GPS_CA_CODE_RATE<> - kappa * state.psrdot.array() / lambda;

  return state;
};

//! ------------------------------------------------------------------------------------------------

// *=== InitFileLoggers ===*
inline FileLoggers InitFileLoggers(SimParam &conf, int &seed) {
  FileLoggers log;
  std::string path = conf.out_folder;
  sturdio::EnsurePathExists(path);
  path += "/" + conf.scenario;
  sturdio::EnsurePathExists(path);
  path += "/" + std::to_string(seed);
  sturdio::EnsurePathExists(path);

  log.navlog.open(path + "/Nav_Results_Log.bin", std::ios::trunc);
  // log.navlog = spdlog::basic_logger_st(
  //     "nav-logger" + std::to_string(seed),
  //     conf.out_folder + "/" + conf.scenario + "/" + std::to_string(seed) +
  //     "/Nav_Results_Log.csv", true);
  // log.navlog->set_pattern("%v");
  // log.navlog->info(
  //     "T [s],ToW [s],Lat [deg],Lon [deg],Alt [m],vN [m/s],vE [m/s],vD [m/s],Roll [deg],Pitch"
  //     "[deg], Yaw[deg],cb [m],cd [m/s]");

  log.errlog.open(path + "/Error_Results_Log.bin", std::ios::trunc);
  // log.errlog = spdlog::basic_logger_st(
  //     "err-logger" + std::to_string(seed),
  //     conf.out_folder + "/" + conf.scenario + "/" + std::to_string(seed) +
  //     "/Error_Results_Log.csv", true);
  // log.errlog->set_pattern("%v");
  // log.errlog->info(
  //     "T [s],ToW [s],Lat [m],Lon [m],Alt [m],vN [m/s],vE [m/s],vD [m/s],Roll [deg],Pitch"
  //     "[deg], Yaw[deg],cb [m],cd [m/s]");

  log.varlog.open(path + "/Variance_Results_Log.bin", std::ios::trunc);
  // log.varlog = spdlog::basic_logger_st(
  //     "var-logger" + std::to_string(seed),
  //     conf.out_folder + "/" + conf.scenario + "/" + std::to_string(seed) +
  //         "/Variance_Results_Log.csv",
  //     true);
  // log.varlog->set_pattern("%v");
  // log.varlog->info(
  //     "T [s],ToW [s],Lat Var [m^2],Lon Var [m^2],Alt Var [m^2],vN Var "
  //     "[(m/s)^2],vE Var [(m/s)^2],vD Var [(m/s)^2],Roll Var [deg^2],Pitch Var [deg^2],Yaw
  //     Var"
  //     "[deg^2],cb Var [m^2],cd Var [(m/s)^2]");

  for (int i = 0; i < conf.n_sv; i++) {
    log.channellogs.push_back(std::ofstream(
        path + "/Channel_" + std::to_string(i) + "_Results_Log.bin", std::ios::trunc));
    // std::string channel = std::to_string(i);
    // log.channellogs.push_back(spdlog::basic_logger_st(
    //     "channel-" + channel + "-logger" + std::to_string(seed),
    //     conf.out_folder + "/" + conf.scenario + "/" + std::to_string(seed) + "/Channel_" +
    //     channel +
    //         "_Results_Log.csv",
    //     true));
    // log.channellogs[i]->set_pattern("%v");
    // if (!conf.is_multi_antenna) {
    //   log.channellogs[i]->info(
    //       "T [s],ToW [s],True Phase [rad],True Omega [rad/s],True Chips [chip],True Chip Rate "
    //       "[chip/s],True CNo,Est Phase [rad],Est Omega [rad/s],Est Chips [chip],Est Chip Rate "
    //       "[chip/s],Est CNo,IE,IP,IL,QE,QP,QL,IP_1,IP_2,QP_1,QP_2,Phase Disc [rad],Freq Disc "
    //       "[m/s],Code Disc [m]");
    // } else {
    //   if (conf.n_ant == 2) {
    //     log.channellogs[i]->info(
    //         "T [s],ToW [s],True Phase [rad],True Omega [rad/s],True Chips [chip],True Chip Rate "
    //         "[chip/s],True CNo,Est Phase [rad],Est Omega [rad/s],Est Chips [chip],Est Chip Rate "
    //         "[chip/s],Est CNo,IE,IP,IL,QE,QP,QL,IP_1,IP_2,QP_1,QP_2,Phase Disc [rad],Freq Disc "
    //         "[m/s],Code Disc [m],IP_A0,QP_A0,IP_A1,QP_A1,Est CNo BS");
    //   } else if (conf.n_ant == 3) {
    //     log.channellogs[i]->info(
    //         "T [s],ToW [s],True Phase [rad],True Omega [rad/s],True Chips [chip],True Chip Rate "
    //         "[chip/s],True CNo,Est Phase [rad],Est Omega [rad/s],Est Chips [chip],Est Chip Rate "
    //         "[chip/s],Est CNo,IE,IP,IL,QE,QP,QL,IP_1,IP_2,QP_1,QP_2,Phase Disc [rad],Freq Disc "
    //         "[m/s],Code Disc [m],IP_A0,QP_A0,IP_A1,QP_A1,IP_A2,QP_A2,Est CNo BS");
    //   } else if (conf.n_ant == 4) {
    //     log.channellogs[i]->info(
    //         "T [s],ToW [s],True Phase [rad],True Omega [rad/s],True Chips [chip],True Chip Rate "
    //         "[chip/s],True CNo,Est Phase [rad],Est Omega [rad/s],Est Chips [chip],Est Chip Rate "
    //         "[chip/s],Est CNo,IE,IP,IL,QE,QP,QL,IP_1,IP_2,QP_1,QP_2,Phase Disc [rad],Freq Disc "
    //         "[m/s],Code Disc [m],IP_A0,QP_A0,IP_A1,QP_A1,IP_A2,QP_A2,IP_A3,QP_A3,Est CNo BS");
    //   }
    // }
  }

  return log;
};

inline void LogResult(
    SimParam &conf,
    FileLoggers &log,
    Truth &truth_k,
    sturdins::Kns &nav,
    std::vector<TrackingStates> &true_state_k,
    TrackingStates &nav_state,
    Eigen::VectorXd &true_cno,
    Eigen::VectorXd &est_cno,
    Eigen::VectorXd &est_cno_bs,
    Eigen::VectorXd &dR,
    Eigen::VectorXd &dRR,
    Eigen::MatrixXd &dP,
    Eigen::Matrix3Xcd &R_bs,
    Eigen::Matrix3Xcd &R1_bs,
    Eigen::Matrix3Xcd &R2_bs,
    Eigen::MatrixXcd &RP) {
  double R2Dsq = navtools::RAD2DEG<> * navtools::RAD2DEG<>;

  // log results
  Eigen::Vector3d rpy_est, lla_true;
  navtools::quat2euler<true, double>(rpy_est, nav.q_b_l_);
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
    double true_cno_db = 10.0 * std::log10(true_cno(i));
    double est_cno_db = 10.0 * std::log10(est_cno(i));
    double est_cno_bs_db = 10.0 * std::log10(est_cno_bs(i));

    // clang-format off
    log.channellogs[i].write(reinterpret_cast<char *>(&truth_k.t), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.ToW(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k[0].phase(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k[0].omega(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k[0].chip(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&true_state_k[0].chip_rate(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&true_cno_db), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.phase(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.omega(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.chip(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&nav_state.chip_rate(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&est_cno_db), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&R_bs(0, i)), sizeof(std::complex<double>));
    log.channellogs[i].write(reinterpret_cast<char *>(&R_bs(1, i)), sizeof(std::complex<double>));
    log.channellogs[i].write(reinterpret_cast<char *>(&R_bs(2, i)), sizeof(std::complex<double>));
    log.channellogs[i].write(reinterpret_cast<char *>(&R1_bs(1, i)), sizeof(std::complex<double>));
    log.channellogs[i].write(reinterpret_cast<char *>(&R2_bs(1, i)), sizeof(std::complex<double>));
    log.channellogs[i].write(reinterpret_cast<char *>(&dP(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&dRR(i)), sizeof(double));
    log.channellogs[i].write(reinterpret_cast<char *>(&dR(i)), sizeof(double));
    // clang-format on

    if (conf.n_ant == 2) {
      // clang-format off
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(0, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(1, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&est_cno_bs_db), sizeof(double));
      // clang-format on
    } else if (conf.n_ant == 3) {
      // clang-format off
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(0, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(1, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(2, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&est_cno_bs_db), sizeof(double));
      // clang-format on
    } else if (conf.n_ant == 4) {
      // clang-format off
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(0, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(1, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(2, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&RP(3, i)), sizeof(std::complex<double>));
      log.channellogs[i].write(reinterpret_cast<char *>(&est_cno_bs_db), sizeof(double));
      // clang-format on
    }
  }

  // log.navlog->info(
  //     "{:.3f},{:.3f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},"
  //     "{:."
  //     "15f},{:.15f}",
  //     truth_k.t,
  //     nav_state.ToW(0),
  //     nav.phi_ * navtools::RAD2DEG<>,
  //     nav.lam_ * navtools::RAD2DEG<>,
  //     nav.h_,
  //     nav.vn_,
  //     nav.ve_,
  //     nav.vd_,
  //     rpy_est(0) * navtools::RAD2DEG<>,
  //     rpy_est(1) * navtools::RAD2DEG<>,
  //     rpy_est(2) * navtools::RAD2DEG<>,
  //     nav.cb_,
  //     nav.cd_);
  // log.errlog->info(
  //     "{:.3f},{:.3f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},"
  //     "{:."
  //     "15f},{:.15f}",
  //     truth_k.t,
  //     nav_state.ToW(0),
  //     ned_err(0),
  //     ned_err(1),
  //     ned_err(2),
  //     truth_k.vn - nav.vn_,
  //     truth_k.ve - nav.ve_,
  //     truth_k.vd - nav.vd_,
  //     (truth_k.roll - rpy_est(0)) * navtools::RAD2DEG<>,
  //     (truth_k.pitch - rpy_est(1)) * navtools::RAD2DEG<>,
  //     (truth_k.yaw - rpy_est(2)) * navtools::RAD2DEG<>,
  //     truth_k.cb - nav.cb_,
  //     truth_k.cd - nav.cd_);
  // log.varlog->info(
  //     "{:.3f},{:.3f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},"
  //     "{:."
  //     "15f},{:.15f}",
  //     truth_k.t,
  //     nav_state.ToW(0),
  //     nav.P_(0, 0),
  //     nav.P_(1, 1),
  //     nav.P_(2, 2),
  //     nav.P_(3, 3),
  //     nav.P_(4, 4),
  //     nav.P_(5, 5),
  //     nav.P_(6, 6) * R2Dsq,
  //     nav.P_(7, 7) * R2Dsq,
  //     nav.P_(8, 8) * R2Dsq,
  //     nav.P_(9, 9),
  //     nav.P_(10, 10));
  // for (int i = 0; i < conf.n_sv; i++) {
  //   if (conf.n_ant == 2) {
  //     log.channellogs[i]->info(
  //         "{:.3f},{:.3f},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{"
  //         "},{},{},{},{},{}",
  //         truth_k.t,
  //         nav_state.ToW(i),
  //         true_state_k[0].phase(i),
  //         true_state_k[0].omega(i),
  //         true_state_k[0].chip(i),
  //         true_state_k[0].chip_rate(i),
  //         10.0 * std::log10(true_cno(i)),
  //         nav_state.phase(i),
  //         nav_state.omega(i),
  //         nav_state.chip(i),
  //         nav_state.chip_rate(i),
  //         10.0 * std::log10(est_cno(i)),
  //         R[0](0, i).real(),
  //         R[0](1, i).real(),
  //         R[0](2, i).real(),
  //         R[0](0, i).imag(),
  //         R[0](1, i).imag(),
  //         R[0](2, i).imag(),
  //         R1[0](1, i).real(),
  //         R2[0](1, i).real(),
  //         R1[0](1, i).imag(),
  //         R2[0](1, i).imag(),
  //         dP(i),
  //         dRR(i),
  //         dR(i),
  //         RP(0, i).real(),
  //         RP(0, i).imag(),
  //         RP(1, i).real(),
  //         RP(1, i).imag(),
  //         10.0 * std::log10(est_cno_bs(i)));
  //   } else if (conf.n_ant == 3) {
  //     log.channellogs[i]->info(
  //         "{:.3f},{:.3f},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{"
  //         "},{},{},{},{},{},{},{}",
  //         truth_k.t,
  //         nav_state.ToW(i),
  //         true_state_k[0].phase(i),
  //         true_state_k[0].omega(i),
  //         true_state_k[0].chip(i),
  //         true_state_k[0].chip_rate(i),
  //         10.0 * std::log10(true_cno(i)),
  //         nav_state.phase(i),
  //         nav_state.omega(i),
  //         nav_state.chip(i),
  //         nav_state.chip_rate(i),
  //         10.0 * std::log10(est_cno(i)),
  //         R[0](0, i).real(),
  //         R[0](1, i).real(),
  //         R[0](2, i).real(),
  //         R[0](0, i).imag(),
  //         R[0](1, i).imag(),
  //         R[0](2, i).imag(),
  //         R1[0](1, i).real(),
  //         R2[0](1, i).real(),
  //         R1[0](1, i).imag(),
  //         R2[0](1, i).imag(),
  //         dP(i),
  //         dRR(i),
  //         dR(i),
  //         RP(0, i).real(),
  //         RP(0, i).imag(),
  //         RP(1, i).real(),
  //         RP(1, i).imag(),
  //         RP(2, i).real(),
  //         RP(2, i).imag(),
  //         10.0 * std::log10(est_cno_bs(i)));
  //   } else if (conf.n_ant == 4) {
  //     log.channellogs[i]->info(
  //         "{:.3f},{:.3f},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{"
  //         "},{},{},{},{},{},{},{},{},{}",
  //         truth_k.t,
  //         nav_state.ToW(i),
  //         true_state_k[0].phase(i),
  //         true_state_k[0].omega(i),
  //         true_state_k[0].chip(i),
  //         true_state_k[0].chip_rate(i),
  //         10.0 * std::log10(true_cno(i)),
  //         nav_state.phase(i),
  //         nav_state.omega(i),
  //         nav_state.chip(i),
  //         nav_state.chip_rate(i),
  //         10.0 * std::log10(est_cno(i)),
  //         R[0](0, i).real(),
  //         R[0](1, i).real(),
  //         R[0](2, i).real(),
  //         R[0](0, i).imag(),
  //         R[0](1, i).imag(),
  //         R[0](2, i).imag(),
  //         R1[0](1, i).real(),
  //         R2[0](1, i).real(),
  //         R1[0](1, i).imag(),
  //         R2[0](1, i).imag(),
  //         dP(i),
  //         dRR(i),
  //         dR(i),
  //         RP(0, i).real(),
  //         RP(0, i).imag(),
  //         RP(1, i).real(),
  //         RP(1, i).imag(),
  //         RP(2, i).real(),
  //         RP(2, i).imag(),
  //         RP(3, i).real(),
  //         RP(3, i).imag(),
  //         10.0 * std::log10(est_cno_bs(i)));
  //   }
  // }
}

//! ------------------------------------------------------------------------------------------------

inline void CalculateDiscriminators(
    const double &T,
    const double &lambda,
    const double &beta,
    const Eigen::Ref<const Eigen::VectorXd> &cno,
    const Eigen::Ref<const Eigen::Matrix3Xcd> &R,
    const Eigen::Ref<const Eigen::Matrix3Xcd> &R1,
    const Eigen::Ref<const Eigen::Matrix3Xcd> &R2,
    Eigen::Ref<Eigen::VectorXd> dR,
    Eigen::Ref<Eigen::VectorXd> dRR,
    Eigen::Ref<Eigen::VectorXd> dR_var,
    Eigen::Ref<Eigen::VectorXd> dRR_var) {
  // discriminators
  // double E, L, X, D, tmp;
  // for (int i = 0; i < dR.size(); i++) {
  //   E = std::norm(R(0, i));
  //   L = std::norm(R(2, i));
  //   dR(i) = beta * 0.5 * (E - L) / (E + L);

  //   X = R1(1, i).real() * R2(1, i).imag() - R2(1, i).real() * R1(1, i).imag();
  //   D = R1(1, i).real() * R2(1, i).real() + R1(1, i).imag() * R2(1, i).imag();
  //   dRR(i) = -lambda * std::atan2(X, D) / (navtools::PI<> * T);

  //   tmp = 1.0 / (cno(i) * T);
  //   dR_var(i) = beta * beta * 0.5 * tmp * (0.5 + tmp);
  //   dRR_var(i) = 2.0 * lambda * lambda / T / T / navtools::PI_SQU<> * tmp * (tmp + 1.0);
  // }
  Eigen::VectorXd E = R.row(0).cwiseAbs2();
  Eigen::VectorXd L = R.row(2).cwiseAbs2();
  Eigen::VectorXd X = R1.row(1).real().array() * R2.row(1).imag().array() -
                      R2.row(1).real().array() * R1.row(1).imag().array();
  Eigen::VectorXd D = R1.row(1).real().array() * R2.row(1).real().array() +
                      R1.row(1).imag().array() * R2.row(1).imag().array();
  dR = beta * 0.5 * (E - L).array() / (E + L).array();
  dRR = X.binaryExpr(
      D, [&](double a, double b) { return -lambda * std::atan2(a, b) / (navtools::PI<> * T); });

  Eigen::VectorXd tmp = 1.0 / (cno * T).array();
  dR_var = beta * beta * 0.5 * tmp.array() * (0.5 + tmp.array());
  dRR_var = 2.0 * lambda * lambda / T / T / navtools::PI_SQU<> * tmp.array() * (tmp.array() + 1.0);
}

inline void CalculateArrayDiscriminators(
    const double &T,
    const double &lamb,
    const double &beta,
    const Eigen::Ref<const Eigen::VectorXd> &cno,
    const Eigen::Ref<const Eigen::VectorXd> &cno_bs,
    const Eigen::Ref<const Eigen::MatrixXcd> &R,       // n_ant x n_sv (only prompt correlators)
    const Eigen::Ref<const Eigen::Matrix3Xcd> &R_bs,   // 3 x n_sv
    const Eigen::Ref<const Eigen::Matrix3Xcd> &R1_bs,  // 3 x n_sv
    const Eigen::Ref<const Eigen::Matrix3Xcd> &R2_bs,  // 3 x n_sv
    Eigen::Ref<Eigen::VectorXd> dR,
    Eigen::Ref<Eigen::VectorXd> dRR,
    Eigen::Ref<Eigen::MatrixXd> dP,
    Eigen::Ref<Eigen::VectorXd> dR_var,
    Eigen::Ref<Eigen::VectorXd> dRR_var,
    Eigen::Ref<Eigen::MatrixXd> dP_var) {
  // beamsteered pseudorange and pseudorange-rate discriminators
  Eigen::VectorXd E = R_bs.row(0).cwiseAbs2();
  Eigen::VectorXd L = R_bs.row(2).cwiseAbs2();
  Eigen::VectorXd X = R1_bs.row(1).real().array() * R2_bs.row(1).imag().array() -
                      R2_bs.row(1).real().array() * R1_bs.row(1).imag().array();
  Eigen::VectorXd D = R1_bs.row(1).real().array() * R2_bs.row(1).real().array() +
                      R1_bs.row(1).imag().array() * R2_bs.row(1).imag().array();

  dR = beta * 0.5 * (E - L).array() / (E + L).array();
  dRR = X.binaryExpr(
      D, [&](double a, double b) { return -lamb * std::atan2(a, b) / (navtools::PI<> * T); });

  Eigen::VectorXd tmp_bs = 1.0 / (cno_bs * T).array();
  dR_var = beta * beta * 0.5 * tmp_bs.array() * (0.5 + tmp_bs.array());
  dRR_var = 2.0 * std::pow(lamb / T / navtools::PI<>, 2) * tmp_bs.array() * (tmp_bs.array() + 1.0);

  // delta phase discriminators
  dP = R.imag().binaryExpr(R.real(), [](double a, double b) { return std::atan2(a, b); });
  Eigen::RowVectorXd dP0 = dP.row(0);
  dP = (-dP).rowwise() + dP0;
  dP = dP.unaryExpr(&navtools::WrapPiToPiFunc<double>);

  Eigen::VectorXd tmp = 1.0 / (cno * T).array();
  dP_var.row(0) = 4.0 * tmp.array() * (0.5 * tmp.array() + 1.0);
  for (int i = 1; i < R.rows(); i++) {
    dP_var.row(i) = dP_var.row(0);
  }
}