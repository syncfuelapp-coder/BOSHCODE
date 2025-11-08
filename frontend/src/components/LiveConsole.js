import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Terminal } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const LiveConsole = ({ logs }) => {
  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="live-console">
      <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <Terminal className="w-5 h-5 text-cyan-400" />
        Live Console
      </h3>
      
      <ScrollArea className="h-48 w-full">
        <div className="space-y-1 font-mono text-sm">
          {logs.length === 0 ? (
            <p className="text-slate-500 italic">No activity yet...</p>
          ) : (
            <AnimatePresence>
              {logs.map((log, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`py-1 ${
                    log.includes("WIN") ? "text-emerald-400" :
                    log.includes("LOSS") ? "text-red-400" :
                    log.includes("ERROR") ? "text-rose-500" :
                    "text-slate-300"
                  }`}
                  data-testid={`console-log-${idx}`}
                >
                  {log}
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>
      </ScrollArea>
    </Card>
  );
};

export default LiveConsole;
