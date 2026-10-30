// Pont entre Hollow Knight et l'IA (Python).
//
// A chaque image, le mod envoie l'etat du jeu en JSON, par UDP, a 127.0.0.1:9999 :
// la scene, le chevalier (position, vitesse, vie, ame, direction, au sol, controlable)
// et le boss (l'ennemi qui a le plus de vie dans la scene : nom, position, vie).
// Il ecoute aussi des commandes sur 127.0.0.1:9998, par exemple « speed 2 » pour
// accelerer le jeu pendant l'entrainement (« speed 1 » pour revenir a la normale).

using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Modding;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKAIBridge
{
    public class HKAIBridge : Mod
    {
        public HKAIBridge() : base("HKAIBridge") { }

        public override string GetVersion() => "0.1.0";

        public override void Initialize()
        {
            var holder = new GameObject("HKAIBridge");
            UnityEngine.Object.DontDestroyOnLoad(holder);
            holder.AddComponent<Bridge>();
            Log("pont vers l'IA pret : etat sur le port " + Bridge.StatePort + ", commandes sur le port " + Bridge.CommandPort);
        }
    }

    public class Bridge : MonoBehaviour
    {
        public const int StatePort = 9999;
        public const int CommandPort = 9998;

        private UdpClient sender;
        private UdpClient listener;
        private IPEndPoint target;
        private float speed = 1f;
        private HealthManager boss;
        private int bossMaxHp;
        private string bossScene = "";
        private float searchAt;

        private void Awake()
        {
            sender = new UdpClient();
            target = new IPEndPoint(IPAddress.Loopback, StatePort);
            listener = new UdpClient(new IPEndPoint(IPAddress.Loopback, CommandPort));
            listener.Client.Blocking = false;
        }

        private void OnDestroy()
        {
            if (sender != null) sender.Close();
            if (listener != null) listener.Close();
        }

        private void Update()
        {
            try
            {
                ReadCommands();
                // The game slows time down for a moment when the knight is hit: leave those alone
                if (speed != 1f && Time.timeScale == 1f) Time.timeScale = speed;
                Send();
            }
            catch (Exception error)
            {
                Debug.Log("[HKAIBridge] " + error);
            }
        }

        private void ReadCommands()
        {
            while (listener.Available > 0)
            {
                IPEndPoint from = null;
                string text = Encoding.UTF8.GetString(listener.Receive(ref from)).Trim();
                string[] parts = text.Split(' ');
                if (parts.Length == 2 && parts[0] == "speed"
                    && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float wanted))
                {
                    speed = Mathf.Clamp(wanted, 0.1f, 10f);
                    Time.timeScale = speed;
                }
            }
        }

        private void FindBoss(string scene)
        {
            if (scene != bossScene)
            {
                bossScene = scene;
                boss = null;
                bossMaxHp = 0;
                searchAt = 0f;
            }
            bool lost = boss == null || !boss.gameObject.activeInHierarchy;
            if (!lost || Time.unscaledTime < searchAt) return;
            searchAt = Time.unscaledTime + 0.5f;
            HealthManager best = null;
            foreach (HealthManager enemy in UnityEngine.Object.FindObjectsOfType<HealthManager>())
            {
                if (enemy.hp > 0 && (best == null || enemy.hp > best.hp)) best = enemy;
            }
            if (best != null && best != boss)
            {
                boss = best;
                bossMaxHp = best.hp;
            }
        }

        private void Send()
        {
            string scene = SceneManager.GetActiveScene().name;
            FindBoss(scene);
            var json = new StringBuilder(320);
            json.Append("{\"t\":").Append(F(Time.unscaledTime));
            json.Append(",\"speed\":").Append(F(speed));
            json.Append(",\"scene\":\"").Append(scene).Append('"');

            HeroController hero = HeroController.instance;
            PlayerData data = PlayerData.instance;
            if (hero != null && data != null)
            {
                Vector3 at = hero.transform.position;
                Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
                Vector2 velocity = body != null ? body.velocity : Vector2.zero;
                json.Append(",\"hero\":{\"x\":").Append(F(at.x)).Append(",\"y\":").Append(F(at.y))
                    .Append(",\"vx\":").Append(F(velocity.x)).Append(",\"vy\":").Append(F(velocity.y))
                    .Append(",\"hp\":").Append(data.health).Append(",\"maxhp\":").Append(data.maxHealth)
                    .Append(",\"soul\":").Append(data.MPCharge)
                    .Append(",\"right\":").Append(B(hero.cState.facingRight))
                    .Append(",\"ground\":").Append(B(hero.cState.onGround))
                    .Append(",\"dead\":").Append(B(hero.cState.dead))
                    .Append(",\"control\":").Append(B(hero.acceptingInput))
                    .Append('}');
            }
            if (boss != null)
            {
                if (boss.hp > bossMaxHp) bossMaxHp = boss.hp;
                Vector3 at = boss.transform.position;
                json.Append(",\"boss\":{\"name\":\"").Append(boss.gameObject.name.Replace("\"", "'"))
                    .Append("\",\"x\":").Append(F(at.x)).Append(",\"y\":").Append(F(at.y))
                    .Append(",\"hp\":").Append(Math.Max(boss.hp, 0)).Append(",\"maxhp\":").Append(bossMaxHp)
                    .Append('}');
            }
            json.Append('}');
            byte[] bytes = Encoding.UTF8.GetBytes(json.ToString());
            sender.Send(bytes, bytes.Length, target);
        }

        private static string F(float value) => value.ToString("0.###", CultureInfo.InvariantCulture);

        private static string B(bool value) => value ? "true" : "false";
    }
}
