import React from 'react';
import { Task, userStudyTasks } from '../lib/tasks';

interface TaskSelectorProps {
    onTaskChange: (selectedTask: Task) => void;
}

export const TaskSelector: React.FC<TaskSelectorProps> = ({ onTaskChange }) => {
    const [selectedTaskId, setSelectedTaskId] = React.useState('');

    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setSelectedTaskId(event.target.value);
        const selectedTask = userStudyTasks.find(task => task.id === event.target.value);
        if (selectedTask) {
            onTaskChange(selectedTask);
        }
    };

    return (
        <select
            onChange={handleChange}
            value={selectedTaskId}
            className='taskSelector'
        >
            <option value="" disabled>Select a task</option>
            {userStudyTasks.map(task => (
                <option key={task.id} value={task.id}>
                    {task.title} - {task.description}
                </option>
            ))}
        </select>
    );
};